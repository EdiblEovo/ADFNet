"""
Encoder for few shot segmentation (VGG16)
"""
import os
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, in_channels=3, pretrained_path=None, depth_enable=True, att_enable=False):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.depth_enable = depth_enable
        self.att_enable = att_enable

        if att_enable:

            #### RGB ENCODER ####
            self.CBR1_RGB_ENC = self._make_layer(2, in_channels, 64)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.CBR2_RGB_ENC = self._make_layer(2, 64, 128)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.CBR3_RGB_ENC = self._make_layer(3, 128, 256)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.CBR4_RGB_ENC = self._make_layer(3, 256, 512)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.CBR5_RGB_ENC = self._make_layer(3, 512, 512, lastRelu=False)

            #### DEPTH ENCODER ####
            self.CBR1_DEPTH_ENC = self._make_layer(2, in_channels, 64)
            self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
            self.CBR2_DEPTH_ENC = self._make_layer(2, 64, 128)
            self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
            self.CBR3_DEPTH_ENC = self._make_layer(3, 128, 256)
            self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
            self.CBR4_DEPTH_ENC = self._make_layer(3, 256, 512)
            self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2)
            self.CBR5_DEPTH_ENC = self._make_layer(3, 512, 512, lastRelu=False)

            self.AFC5_RGB = self._make_afc_module(512)
            self.AFC5_DEPTH = self._make_afc_module(512)

            
        else:
            self.features = nn.Sequential(
                self._make_layer(2, in_channels, 64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layer(2, 64, 128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layer(3, 128, 256),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layer(3, 256, 512),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self._make_layer(3, 512, 512, lastRelu=False),
            )
        
        self._init_weights()

    def forward(self, rgb_inputs, depth_inputs):

        if self.att_enable:

            # Stage 1
            x = self.CBR1_DEPTH_ENC(depth_inputs)
            y = self.CBR1_RGB_ENC(rgb_inputs)

            x = self.pool1_d(x)
            y = self.pool1(y)

            # Stage 2
            x = self.CBR2_DEPTH_ENC(x)
            y = self.CBR2_RGB_ENC(y)

            x = self.pool2_d(x)
            y = self.pool2(y)

            # Stage 3
            x = self.CBR3_DEPTH_ENC(x)
            y = self.CBR3_RGB_ENC(y)

            x = self.pool3_d(x)
            y = self.pool3(y)

            # Stage 4
            x = self.CBR4_DEPTH_ENC(x)
            y = self.CBR4_RGB_ENC(y)

            x = self.pool4_d(x)
            y = self.pool4(y)

            # Stage 5
            x = self.CBR5_DEPTH_ENC(x)
            y = self.CBR5_RGB_ENC(y)

            y_1 = self.AFC5_RGB(y)
            x_1 = self.AFC5_DEPTH(x)
            y = y * y_1 + x * x_1
            
        else:
            y = self.features(rgb_inputs)

        return y

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _make_afc_module(self, in_channels):
        layer = []
        layer.append(nn.AdaptiveAvgPool2d((1,1)))
        layer.append(nn.Conv2d(in_channels, in_channels, kernel_size=1,
                                dilation=1, padding=0))
        layer.append(nn.Sigmoid())
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            rgb_dic = torch.load(os.path.join(f'{self.pretrained_path}_rgb/', 'fcn32s_vgg16_pascal_voc.pth'), map_location='cpu')
            dep_dic = torch.load(os.path.join(f'{self.pretrained_path}_dep/', 'fcn32s_vgg16_pascal_voc.pth'), map_location='cpu')
            rgb_keys = list(rgb_dic.keys())
            dep_keys = list(dep_dic.keys())

            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(26):
                new_dic[new_keys[i]] = rgb_dic[rgb_keys[i]]
                print(new_keys[i])
                print(rgb_keys[i])
                if self.att_enable:
                    new_dic[new_keys[i+26]] = dep_dic[dep_keys[i]]
                    print(new_keys[i+26])
                    print(dep_keys[i])

            self.load_state_dict(new_dic)
            if self.att_enable:
                for param in self.parameters():
                    param.requires_grad = False
                for name, value in self.named_parameters():
                    if 'AFC' in name:
                        value.requires_grad = True
                        print(name)
