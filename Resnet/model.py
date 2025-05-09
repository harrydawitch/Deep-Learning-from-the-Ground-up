import torch
import torch.nn as nn

class resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, first_block = False, down_block= False):
        super(resnet_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_layers = nn.ModuleList()

        down_layer = True
        for _ in range(num_conv):
            layers = self.setup_layers(in_channels, out_channels, first_block, down_block, down_layer)
            self.conv_layers.append(nn.Sequential(*layers))
            down_layer = False

            
    def setup_layers(self, in_channels, out_channels, first_block= False, down_block= False, down_layer= False):
        layers= []
        if first_block:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias= False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace= True))
        
        elif down_block:
            layers.append(nn.Conv2d(in_channels//2 if down_layer else in_channels, out_channels, 3, 2 if down_layer else 1, 1, bias= False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace= True))
            
        return layers
        
    def forward(self, x, identity):
                
        for idx, seq in enumerate(self.conv_layers):
            if idx % 2 == 0:
                x = seq(x)
                identity= self.shape_checking(x, identity)
            else:
                for layer in seq:
                    if isinstance(layer, nn.ReLU):
                        x += identity
                    x = layer(x)
                        
                identity = x
        
        return identity
    
    
    def shape_checking(self, x, identity):
        x_c = x.shape[1]
        i_c = identity.shape[1]
        
        if i_c != x_c: # If identity channels aren't equal to input channels
            identity = nn.Conv2d(in_channels= i_c, out_channels= x_c, kernel_size= 1, stride= 2, padding= 0, bias= False)(identity)

            
        return identity
        

class resnet34(nn.Module):
    def __init__(self, in_channels= 3, out_channels= 512):
        super(resnet34, self).__init__()
        
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size= 7, stride= 2, padding= 3, bias= False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
                                        )
        
        self.last_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size= 1),
            nn.Flatten(),
            nn.Linear(out_channels, 1000)
                                       )
        
        self.resnet_block = nn.ModuleList()
        
        self.resnet_block.append(resnet_block(64, 64, 6, first_block= True))
        self.resnet_block.append(resnet_block(128, 128, 8, down_block= True))
        self.resnet_block.append(resnet_block(256, 256, 12, down_block= True))
        self.resnet_block.append(resnet_block(512, 512, 6, down_block= True))


            
    def forward(self, x):
        x = self.first_block(x)
        identity = x
        
        for block in self.resnet_block:
            x = block(x, identity)
            identity = x
        
        x = self.last_block(x)
        return x
        
        
def test():
    x = torch.randn((1, 3, 224, 224))
    model = resnet34()
    preds = model(x)
    print(x.shape)
    print(preds.shape)

if __name__ == "__main__":
    test()