import torch
from torch.nn.parameter import Parameter

class BaseLayer(torch.nn.Module):
    def __init__(self, inputs, outputs, **kwargs):
        torch.nn.Module.__init__(self)
        
        self.n_inputs = inputs
        self.n_outputs = outputs
        
        self.clamp_min = 0 if "clamp_min" not in kwargs else kwargs["clamp_min"]
        self.clamp_max = 10 if "clamp_max" not in kwargs else kwargs["clamp_max"]
        
    def clamp(self, x):
        def norm_sigmoid(x, min, max):
            return  1 / (torch.exp(-((max-min)*x+min)) + 1)
        if self.clamp_min is not None and self.clamp_max is not None:
            return norm_sigmoid((x-self.clamp_min)/(self.clamp_max-self.clamp_min),-2,2)*(self.clamp_max-self.clamp_min)+self.clamp_min
        elif self.clamp_min is not None and self.clamp_max is None:
            return self.clamp_min + torch.nn.functional.softplus(x-self.clamp_min)
        elif self.clamp_min is None and self.clamp_max is not None:
            return -(self.clamp_max + torch.nn.functional.softplus(-x+self.clamp_max))
        else:
            return 1*x
            
    def reset(self):
        raise NotImplementedError("You should implement a reset function in a BaseLayer")
    
    def to(self, *args, **kwargs):
        self = torch.nn.Module.to(self, *args, **kwargs) 
        
        return self