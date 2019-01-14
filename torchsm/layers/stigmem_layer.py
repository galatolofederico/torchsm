import torch
from .base_layer import BaseLayer

class StigmergicMemoryLayer(BaseLayer):
    def __init__(self, inputs, space_dim, **kwargs):
        BaseLayer.__init__(self, inputs, space_dim)

        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else 0
        self.hidden_dim = kwargs["hidden_dim"] if "hidden_layers" in kwargs else None

        self.n_inputs = inputs
        self.space_dim = self.n_outputs

        self.init_space = torch.zeros(self.space_dim)

        self.mark_net = torch.nn.Sequential()
        self.tick_net = torch.nn.Sequential()

        if self.hidden_layers != 0:
            self.mark_net.add_module("input_w", torch.nn.Linear(self.n_inputs, self.hidden_dim))
            self.tick_net.add_module("input_w", torch.nn.Linear(self.n_inputs, self.hidden_dim))
            
            self.mark_net.add_module("input_s", torch.nn.PReLU())
            self.tick_net.add_module("input_s", torch.nn.PReLU())
            
            for i in range(0, self.hidden_layers-1):
                self.mark_net.add_module("l"+str(i)+"_w", torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                self.tick_net.add_module("l"+str(i)+"_w", torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                
                self.mark_net.add_module("l"+str(i)+"_s", torch.nn.PReLU())
                self.tick_net.add_module("l"+str(i)+"_s", torch.nn.PReLU())
            
            self.mark_net.add_module("output_w", torch.nn.Linear(self.hidden_dim, self.space_dim))
            self.tick_net.add_module("output_w", torch.nn.Linear(self.hidden_dim, self.space_dim))

            self.mark_net.add_module("output_relu", torch.nn.ReLU())
            self.tick_net.add_module("output_relu", torch.nn.ReLU())
        else:
            self.mark_net.add_module("linear", torch.nn.Linear(self.n_inputs, self.space_dim))
            self.tick_net.add_module("linear", torch.nn.Linear(self.n_inputs, self.space_dim))

            self.mark_net.add_module("output_relu", torch.nn.ReLU())
            self.tick_net.add_module("output_relu", torch.nn.ReLU())
            
        self.reset()

    def forward(self, input_mark, input_tick = None):
        mark = self.mark_net(input_mark)
        tick = self.tick_net(input_tick if input_tick is not None else input_mark)
        self.space = self.clamp(self.space + mark - tick)
        return self.space

    def reset(self):
        self.space = self.init_space.clone()

    def to(self, *args, **kwargs):
        self = BaseLayer.to(self, *args, **kwargs)
        
        self.space = self.space.to(*args, **kwargs)
        self.init_space = self.init_space.to(*args, **kwargs)
        self.mark_net = self.mark_net.to(*args, **kwargs)
        self.tick_net = self.tick_net.to(*args, **kwargs)

        return self
