import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.func as fc
import math

class jvp_MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=[128, 128, 128], method='weight_perturb'):
        super().__init__()
        self.method = method
        self.net = nn.ModuleList([jvp_linear(in_size, hidden_size[0]), jvp_relu()])
        for i in range(len(hidden_size)-1):
            self.net.append(jvp_linear(hidden_size[i], hidden_size[i+1]))
            self.net.append(jvp_relu())
        self.net.append(jvp_linear(hidden_size[-1], out_size))
        self.net.append(jvp_loss())
        self.linear_layers = [layer for layer in self.net if isinstance(layer, jvp_linear)]

        self.output_gradients = {}
        self._register_hooks()

    # log backprop gradients for testing
    def _register_hooks(self):
        def save_gradient_hook(module, grad_input, grad_output):
            self.output_gradients[module] = grad_output[0]  # Store the gradient of the output
        # Register the hook on each linear layer
        for layer in self.net:
            if isinstance(layer, jvp_linear):
                layer.register_backward_hook(save_gradient_hook)

    def forward(self, x, return_loss=False): # will not go through loss fn
        for layer in self.net[:-1]:
            x = layer(x)
        return x

    def prepare_layers(self, x):
        # generate randn guesses at appropriate layers for each method
        # set other required attributes e.g layer.W_next for W^T
        # set layer fwd behaviour (track weight jvps or activation jvps)
        for l, layer in enumerate(self.linear_layers):
            layer.bias_guess = torch.randn_like(layer.bias)
            if self.method == 'weight_perturb':
                layer.weight_guess = torch.randn_like(layer.weight)
                layer.fwd_method = 'weight_perturb'
            elif self.method == 'act_perturb':
                layer.s_guess = torch.randn((x.shape[0], layer.weight.shape[0])) # B x out
                layer.fwd_method = 'act_perturb'
            elif self.method == 'W^T':
                layer.fwd_method = 'act_perturb'
                layer.s_guess = torch.randn((x.shape[0], layer.weight.shape[0])) # B x out
                if l == 0: # input layer does not need activation guess
                    layer.s_guess = torch.zeros(x.shape[0], layer.weight.shape[0])
                if l < len(self.linear_layers)-1:
                    # set W_next for all layers except last
                    layer.W_next = self.linear_layers[l+1].weight
            elif self.method == 'layer_downstream':
                layer.fwd_method = 'act_perturb'
                layer.bias_guess = torch.zeros_like(layer.bias)
                layer.s_guess = torch.zeros(x.shape[0], layer.weight.shape[0])
                if l == len(self.linear_layers)-1: # only guess at last layer
                    layer.s_guess = torch.randn((x.shape[0], layer.weight.shape[0])) # B x out
                    layer.bias_guess = torch.randn_like(layer.bias)

    def jvp_forward(self, x, y):
        self.prepare_layers(x)
        jvp_in = torch.zeros_like(x)
        for layer in self.net[:-1]:
            x, jvp_in = layer.jvp_forward(x, jvp_in)
        loss_layer = self.net[-1]
        loss_layer.y = y
        loss, jvp_out = loss_layer.jvp_forward(x, jvp_in)
        return loss, jvp_out

    def set_grad(self, jvp): # set weight & bias grads for each linear layer
        for l, layer in enumerate(self.linear_layers):
            layer.bias.grad = jvp.mean() * layer.bias_guess
            if self.method == 'weight_perturb':
                layer.weight.grad = jvp.mean() * layer.weight_guess
            elif self.method == 'act_perturb':
                layer.weight.grad = (
                    ((jvp.unsqueeze(-1) * layer.s_guess).unsqueeze(2) * layer.x_in.unsqueeze(1))
                ).sum(dim=0)
                # with torch.enable_grad():
                #    s = F.linear(self.x_in, self.weight, self.bias)
                #    s.backward(gradient=jvp.unsqueeze(-1)*self.s_guess)
            elif self.method == 'W^T':
                if l < len(self.linear_layers)-1:
                    s_next_guess = self.linear_layers[l+1].s_guess
                    dL_ds_i = ((jvp.unsqueeze(-1) * s_next_guess) @ layer.W_next) * (
                                (layer.x_in @ layer.weight.T) > 0)  # B x in
                    layer.weight.grad = (dL_ds_i.unsqueeze(2) * layer.x_in.unsqueeze(1)).mean(dim=0)
                elif l == len(self.linear_layers)-1: # last layer - same as act perturb
                    layer.weight.grad = (
                        ((jvp.unsqueeze(-1) * layer.s_guess).unsqueeze(2) * layer.x_in.unsqueeze(1))
                    ).mean(dim=0)

class jvp_linear(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fwd_method = None
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def act_fwd(self, x, bias): # act perturb
        return F.linear(x, self.weight, bias)

    def weight_fwd(self, x, bias, weight): # track weight jvps for weight perturb
        return F.linear(x, weight, bias)

    def jvp_forward(self, x_in, jvp_in):
        self.x_in = x_in # store x_in for gradient computation
        if self.fwd_method == 'weight_perturb':
            out, jvp_out = fc.jvp(self.weight_fwd, (x_in, self.bias, self.weight), (jvp_in, self.bias_guess, self.weight_guess))
        elif self.fwd_method == 'act_perturb':
            out, jvp = fc.jvp(self.act_fwd, (x_in, self.bias), (jvp_in, self.bias_guess))
            jvp_out = jvp+self.s_guess
        return out, jvp_out

class jvp_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.y = None

    def forward(self, x, y):
        return self.loss_fn(x, y)

    def func(self, x):
        return self.loss_fn(x, self.y)

    def jvp_forward(self, x_in, jvp_in):
        out, jvp_out = fc.jvp(self.func, (x_in,), (jvp_in,))
        return out, jvp_out


class jvp_relu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)

    def func(self, x):
        return F.relu(x)

    def jvp_forward(self, x_in, jvp_in):
        out, jvp_out = fc.jvp(self.forward, (x_in,), (jvp_in,))
        return out, jvp_out
