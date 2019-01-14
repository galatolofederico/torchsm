# torchsm
pytorch implementation of the **Stigmergic Memory** as presented in the paper [Using stigmergy as a computational memory in the design of recurrent neural networks]().

You can use this package to **easly** integrate our model into existing ones

You can safely **mix** native pytorch Modules with ours.  

But **do not forget** to `reset()` them before starting every new time sequence

Implementing our [proposed architecture to solve MNIST]() becomes as easy as:
```python
import torch
import torchsm

net = torchsm.Sequential(
    torchsm.RecurrentStigmergicMemoryLayer(28, 15, hidden_layers=1, hidden_dim=20),
    torch.nn.Linear(15, 10),
    torch.nn.PReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.PReLU()
)
```

You can train the time-unfolded model by computing the loss function on the desired temporal output

```python
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
loss_fn = torch.nn.MSELoss()

for i in range(0,N):
    for X, Y in zip(dataset_X, dataset_Y):
        net.reset()
        out = None
        for i in range(0, X.shape[1]):
            out = net(torch.tensor(X[:,i], dtype=torch.float32))
        
        loss = loss_fn(out, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Does it support batch inputs?

Yes! The inputs have to be batched

```python
for t in range(0, num_ticks):
    batch_out[0], batch_out[1], ... = net(torch.tensor([batch_in[0][t], batch_in[1][t], ...]))
```
### Can it run on CUDA?

Yes and as you will expect from a pytorch Module!  
You just need to call the `to(device)` method on a model to move it in the GPU memory

```python
device = torch.device("cuda")

net = net.to(device)

net(torch.tensor(..., device=device))
```
## Documentation

### torchsm.Sequential

Wrapper of `torch.nn.Sequential` that adds the `reset()` method and forward the call to each `torchsm.BaseLayer` child.

If you want to use a `SequentialContaier` to build your models with one or more torchsm's layers you have to use `torchsm.Sequential` instead of `torch.nn.Sequential` in order to be able to `reset()` them.

### torchsm.StigmergicMemoryLayer

This layer has two hidden ANNs with the layer's inputs as inputs and which outputs respectively determine the marks and ticks of a multi-monodimensional stigmergic space.

![Imgur](https://i.imgur.com/yS4M4nA.png)

### torchsm.RecurrentStigmergicMemoryLayer

This layer is a StigmergicMemoryLayer which output is normalized by a linear layer and recurrently forwarded as input to the two hidden ANNs 

![Imgur](https://i.imgur.com/JWQF6ft.png)

## Citing

We can't wait to see what you will build with torchsm!  
When you will publish your work you can use this BibTex to cite us :)

```
@article{galatolo_snn
,	author	= {Galatolo, Federico A and Cimino, Mario GCA and Vaglini, Gigliola}
,	title	= {Using stigmergy as a computational memory in the design of recurrent neural networks}
,	journal	= {ICPRAM 2019}
,	year	= {2019}
,	pages	= {}
}
```

## Contributing

This code is released under GNU/GPLv3 so feel free to fork it and submit your changes, every PR helps.  
If you need help using it or for any question please reach me at [federico.galatolo@ing.unipi.it](mailto:galatolo.federico@gmail.com) or on Telegram  [@galatolo](https://t.me/galatolo)