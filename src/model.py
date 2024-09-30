import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundModel(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_size ) -> None:
        super().__init__()
        
        self.convBlock1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(hidden_size)
        )
        
        self.convBlock2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(hidden_size*2)
        )
        
        self.convBlock3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size*2, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(hidden_size*4)
        )
        
        self.fc1 = nn.Linear(in_features=384, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
        self.init_weights()
        
    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
            
    def init_weights(self):
        print("Initializing weights for the model with Kaiming Normal")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
# if __name__ == "__main__":
#     import dataset as ds
#     test_dataset, classes = ds.get_dataloader('Signals/test', batch_size=32, shuffle=False,num_workers=4, pin_memory=True, transform=None)
#     input_batch, target_batch = next(iter(test_dataset))
    
#     model = SoundModel(input_shape=1, num_classes=len(classes), hidden_size=16)
#     preds = model(input_batch.unsqueeze(1))
#     print(preds.shape)
#     print(preds)