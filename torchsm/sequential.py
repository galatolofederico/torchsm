import torch
from .layers import BaseLayer

class Sequential(torch.nn.Sequential):
    def __init__(self, *largs, **kwargs):
        torch.nn.Sequential.__init__(self, *largs, **kwargs)

    def reset(self):
        for layer in self.children():
            if isinstance(layer, BaseLayer):
                layer.reset()

    def to(self, *args, **kwargs):
        for layer in self.children():
            layer.to(*args, **kwargs)
        
        return self
