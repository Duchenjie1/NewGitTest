import math
import torch
import torch.nn as nn

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])  #16
    if 'top' in method:  #7x7 frequency space
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]  #mapper_x:[0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2]
        mapper_y = all_top_indices_y[:num_freq]  #mapper_y:[0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction  #16
        self.dct_h = dct_h   #56
        self.dct_w = dct_w   #56

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)  #mapper_x和mapper_y都是固定值
        self.num_split = len(mapper_x)  #16
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] #映射到特征图空间(56*56)位置
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        '''
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        '''

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))  #adaptive_avg_pool2d: resize
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        DCT_features = self.dct_layer(x_pooled).view(n, c, 1, 1)

        #y = self.fc(y).view(n, c, 1, 1)
        return DCT_features


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)  #16

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))  #调用register_buffer()函数
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        # self.register_parameter('weight', nn.Parameter(self.get_dct_filter(height, width, mapper_x, mapper_y, channel))) # du add
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight  #x:[4,128,56,56], weight:[128,56,56]

        result = torch.sum(x, dim=[2,3]) #减少维度2，3
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) #freq:mapper_x, mapper_y
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)  #dct_filter.shape:[64,56,56]

        c_part = channel // len(mapper_x)  #c_part:4，相当于分组卷积的groups=4
        #相当于论文中u，v的值：mapper_x:[0,0,48,0，0，8，8，32，40，8，24，0,0,0,24，16], mapper_y:[0，8，0, 40，16，0，16，0，0，48，0，32，48，24，40，16]
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)): #i:0~15(把X分为15个部分),
            for t_x in range(tile_size_x): #t_x:0~55
                for t_y in range(tile_size_y): #t_y:0~55
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y) #i * c_part: (i+1)*c_part:拼接，[0：4],[4，8],...,[60：64], build_filter:计算特征图每一个位置的DCT
                        
        return dct_filter