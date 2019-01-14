import torch

from .base_layer import BaseLayer
from .stigmem_layer import StigmergicMemoryLayer

class RecurrentStigmergicMemoryLayer(BaseLayer):
    def __init__(self, input, output, **kwargs):
        BaseLayer.__init__(self, input, output, **kwargs)
        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else 0
        self.hidden_dim = kwargs["hidden_dim"] if "hidden_dim" in kwargs else 30
        self.stig_dim = output

        self.stigmem = StigmergicMemoryLayer(
            input+self.stig_dim,
            self.stig_dim,
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim
        )

        self.normalization_layer_mark = torch.nn.Linear(self.stig_dim, self.stig_dim)
        self.normalization_layer_tick = torch.nn.Linear(self.stig_dim, self.stig_dim)

        self.init_recurrent = torch.zeros(1, self.stig_dim)
        self.reset()
    
    def forward(self, input):
        self.recurrent = self.stigmem(
            torch.cat(
                (input, self.normalization_layer_mark(self.recurrent.expand(input.shape[0], self.stig_dim)))
            ,1),
            torch.cat(
                (input, self.normalization_layer_tick(self.recurrent.expand(input.shape[0], self.stig_dim)))
            ,1),
        )
        return self.recurrent
    
    def reset(self):
        self.recurrent = self.init_recurrent.clone()
        self.stigmem.reset()

    def to(self, *args, **kwargs):
        self = BaseLayer.to(self, *args, **kwargs)
        
        self.stigmem = self.stigmem.to(*args, **kwargs)
        self.normalization_layer_mark = self.normalization_layer_mark.to(*args, **kwargs)
        self.normalization_layer_tick = self.normalization_layer_tick.to(*args, **kwargs)

        self.init_recurrent = self.init_recurrent.to(*args, **kwargs)
        self.recurrent = self.recurrent.to(*args, **kwargs)
        return self

