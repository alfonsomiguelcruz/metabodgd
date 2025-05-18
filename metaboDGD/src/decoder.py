import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.nn = nn.ModuleList()

        self.output_layer = None
    
    
    def forward(self, z):
        pass

    @classmethod
    def load(cls):
        pass