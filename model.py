from torch import nn

class Model(nn.Module):
    def __init__(self, in_features, out_features=7):
        super(Model, self).__init__()
        
        # Define the flowstats layers
        self.fc = nn.Sequential(
            
            # Input layer
            nn.Linear(in_features=in_features, out_features=in_features*2, bias=False),
            
            # Hidden layer 1
            nn.BatchNorm1d(in_features*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.15, inplace=False),
            
            # Hidden layer 2
            nn.Linear(in_features=in_features*2, out_features=in_features*3, bias=False),
            nn.BatchNorm1d(in_features*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.15, inplace=False),
            
            # Hidden layer 3
            nn.Linear(in_features=in_features*3, out_features=in_features*3, bias=False),
            nn.BatchNorm1d(in_features*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.15, inplace=False),
            
            # Hiddel layer 4
            nn.Linear(in_features=in_features*3, out_features=in_features*4, bias=False),
            nn.BatchNorm1d(in_features*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.15, inplace=False),
            
            # Output layer
            nn.Linear(in_features=in_features*4, out_features=out_features, bias=True)
        )

    def forward(self, x):
        return self.fc(x)