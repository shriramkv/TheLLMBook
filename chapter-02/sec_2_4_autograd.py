import torch

x, y = torch.tensor(2.0), torch.tensor(1.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

a = torch.sigmoid(w * x + b)
loss = -torch.log(a)
loss.backward()                   # backpropagation in one line
print(w.grad, b.grad)             # tensor(-0.5379) tensor(-0.2689)
