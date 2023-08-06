from torch import nn

class SimpleLinear(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units*2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units*2, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)