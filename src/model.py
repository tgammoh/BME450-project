import torch 
import torch.nn as nn
from config import NUM_CLASSES, NUM_CHANNELS, WINDOW_SIZE, DROPOUT



class EMGNet(nn.Module):


    def __init__(self):
        super(EMGNet, self).__init__()
        

        self.conv1 = nn.Conv1d(in_channels=NUM_CHANNELS, out_channels=32, kernel_size=3, padding=1)
     
        self.bn1 = nn.BatchNorm1d(32) 
        """
        I added this after increasing the conv layers, but the graph for loss and accuracy was ver jumpy 
        and there was a big gap between the training and the testing data, so I added batch normalization.
        """
        self.relu1 = nn.ReLU()
        # self.drop1 = nn.Dropout(p=DROPOUT)




        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
      

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
      


        
        self.pool  = nn.AdaptiveAvgPool1d(output_size=1)

        self.dropout = nn.Dropout(p=DROPOUT) # forcing it to learn without relying on any single feature 


   

        #classifier
        self.fc1 = nn.Linear(128 , NUM_CLASSES)

       


    def forward(self, x):

        x = x.permute(0, 2, 1) # to change from matlab to torch, it has to be [batch, 12 channels, time samples] ]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        #x = self.drop3(x)


        
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)   
        x = self.fc1(x)

        return x


if __name__ == '__main__':
    model = EMGNet()
    print(model)

    dummy = torch.randn(32, WINDOW_SIZE, NUM_CHANNELS)
    output = model(dummy)
    print(f'\nInput shape:  {dummy.shape}')
    print(f'Output shape: {output.shape}')   

    torch.onnx.export(model, dummy, 'emgnet.onnx')
    print('Model exported to emgnet.onnx')