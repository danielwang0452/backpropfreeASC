import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.func as fc
import math

device = 'cpu'
class MLP_backprop(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=[128, 128, 128]):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(in_size, hidden_size[0]),
                                  nn.ReLU()])
        for i in range(len(hidden_size)-1):
            self.net.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, x):
        out = x
        i = 1
        j = 0
        for l, layer in enumerate(self.net):
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                setattr(self, f'x_{i}', out)
                i += 1
            else:
                setattr(self, f's_{j}', out)
                j += 1
        return out

class jvp_MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=[128, 128, 128, 128]):
        super().__init__()
        self.in_layer = jvp_in_layer(in_size, hidden_size[0])
        setattr(self.in_layer, 'layer_num', 0)
        self.hidden_layers = nn.ModuleList([])
        for i in range(len(hidden_size)-1):
            self.hidden_layers.append(jvp_LinearReLU(hidden_size[i], hidden_size[i+1]))
            setattr(self.hidden_layers[-1], 'layer_num', i+1)
        self.out_layer = jvp_out_layer(hidden_size[-1], out_size)
        setattr(self.out_layer, 'layer_num', len(hidden_size))
        #self.optimiser = torch.optim.SGD(self.parameters(), lr=2e-4)
        self.in_size, self.out_size, self.hidden_size = in_size, out_size, hidden_size
        self.num_params = 0
        self.num_params += in_size * hidden_size[0] + hidden_size[0] + hidden_size[-1] * out_size + out_size
        for h in range(len(hidden_size) - 1):
            self.num_params += hidden_size[h] * hidden_size[h + 1] + hidden_size[h + 1]
        self.num_params = 1#torch.sqrt(torch.tensor(self.num_params))
        print(f'num_params:{self.num_params}')
        self.in_layer.num_params = self.num_params
        self.out_layer.num_params = self.num_params

    def jvp_forward(self, x, y, return_loss=True):
        out, jvp_out = self.in_layer.jvp_forward(x, w_next=self.hidden_layers[0].weight)
        for l, layer in enumerate(self.hidden_layers):
            layer.num_params = self.num_params
            if l + 1 < len(self.hidden_layers):
                out, jvp_out = layer.jvp_forward(out, jvp_out, w_next=self.hidden_layers[l + 1].weight)
            else:
                out, jvp_out = layer.jvp_forward(out, jvp_out, w_next=self.out_layer.weight)
        if return_loss:
            loss, jvp_out = self.out_layer.jvp_forward(out, jvp_out, y)
            return loss, jvp_out
        else:
            pred = self.out_layer.jvp_forward(out, jvp_out, y, return_loss=return_loss)
            return pred

    def set_grad(self, jvp):
        self.in_layer.set_grad(jvp)
        for layer in self.hidden_layers:
            layer.set_grad(jvp)
        self.out_layer.set_grad(jvp)
        return

class BackpropLayer(nn.Module):
    def __init__(self, W_i, b_i, W_next, guess):
        super().__init__()
        self.W_i = nn.Parameter(W_i, requires_grad=True)  # Ensure grad computation
        self.b_i = b_i#nn.Parameter(b_i.clone().detach(), requires_grad=True)
        self.W_next = W_next#nn.Parameter(W_next.clone().detach(), requires_grad=True)  # Optional: not updated
        self.guess = guess#nn.Parameter(guess, requires_grad=True)  # Gradient to propagate
        #print(self.b_i)

    def forward(self, x_i):
        # Forward pass
        self.W_i.grad = None
        x_next = F.relu(F.linear(x_i, weight=self.W_i, bias=self.b_i))  # ReLU activation
        #print(x_next.shape, self.guess.shape)
        #s_next = F.linear(x_next, weight=self.W_next, bias=None)        # Next state
        x_next.backward(gradient=self.guess)  # Propagate gradient through computation graph
        return self.W_i.grad  # Return the gradient of W_i


class jvp_in_layer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias_guess = None
        self.weight_guess = None

    def forward(self, x):
        return F.relu(F.linear(x, self.weight, self.bias))

    def fwd_grad(self, params, buffers, names, x):
        out = fc.functional_call(self, ({k: v for k, v in zip(names, params)}, buffers), (x,))
        return out

    def guess_W(self, W_next, x_i):
        #guess = torch.randn((x_i.shape[0], W_next.shape[0])).to(device)  # B x out
        #guess = torch.load('grad_ss.pth')[self.layer_num]
        #left = (guess @ W_next) * ((x_i @ self.weight.T) > 0)  # B x in
        #out = left.unsqueeze(2) * x_i.unsqueeze(1)
        #return out.sum(dim=0)#*(torch.randn_like(self.weight).norm()/out.sum(dim=0).norm())

        # testing 1 layer downstream
        #guess = torch.randn((x_i.shape[0], W_next.shape[1]))#/(torch.tensor(x_i.shape[0]*W_next.shape[1])).sqrt()  # B x out
        #guess = torch.load('grad_xs.pth')[self.layer_num]
        #tensors = [self.weight, self.bias, W_next, guess, x_i]
        #with torch.enable_grad():
        #    grad = backprop(tensors)
        #return grad

        # random guess
        return torch.randn_like(self.weight)

    def jvp_forward(self, x_in, w_next):
        # get params & buffers
        named_buffers = dict(self.named_buffers())
        named_params = dict(self.named_parameters())
        names = named_params.keys()
        params = named_params.values()
        # set tangents
        self.weight_guess = self.guess_W(w_next, x_in)/self.num_params
        self.bias_guess = torch.randn_like(self.bias)/self.num_params
        v_params = [
            self.bias_guess,  # bias
            self.weight_guess  # weight
        ]
        v_params = tuple(v_params)
        # print(v_params)

        f = partial(
            self.fwd_grad,
            names=names,
            buffers=named_buffers,
            x=x_in
        )
        out, jvp_out = fc.jvp(f, (tuple(params),), (v_params,))

        return out, jvp_out  # same shape

    def set_grad(self, jvp_val):
        self.weight.grad = jvp_val * self.weight_guess
        self.bias.grad = jvp_val * self.bias_guess

class jvp_LinearReLU(nn.Module):
    def __init__(self, in_size, out_size, x_param=True):
        super().__init__()
        #f = f_module(in_size, out_size)
        self.x_param = x_param
        if x_param:
            self.x = nn.Parameter(torch.randn(in_size))
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias_guess = None
        self.weight_guess = None

    def forward(self, dummy_var): # dummy var since we treat input as a parameter
        if self.x_param:
            return F.relu(F.linear(self.x, self.weight, self.bias))
        else:
            return F.relu(F.linear(dummy_var, self.weight, self.bias))

    def fwd_grad(self, params, buffers, names):
        out = fc.functional_call(self, ({k: v for k, v in zip(names, params)}, buffers), args=None)
        return out

    def guess_W(self, W_next, x_i):
        #guess = torch.randn((x_i.shape[0], W_next.shape[0])).to(device)  # B x out
        #guess = torch.load('grad_ss.pth')[self.layer_num]
        #left = (guess @ W_next) * ((x_i @ self.weight.T) > 0)  # B x in
        #out = left.unsqueeze(2) * x_i.unsqueeze(1)
        #return out.sum(dim=0)##*(torch.randn_like(self.weight).norm()/out.sum(dim=0).norm())

        # testing 1 layer downstream
        #guess = torch.randn((x_i.shape[0], W_next.shape[1]))#/(torch.tensor(x_i.shape[0]*W_next.shape[1])).sqrt()  # B x out
        #guess = torch.load('grad_xs.pth')[self.layer_num]
        #tensors = [self.weight, self.bias, W_next, guess, x_i]
        #with torch.enable_grad():
        #    grad = backprop(tensors)
        #return grad

        # random guess
        return torch.randn_like(self.weight)

    def jvp_forward(self, x_in, jvp_in, w_next):
        # get params & buffers
        named_buffers = dict(self.named_buffers())
        named_params = dict(self.named_parameters())
        named_params['x'] = x_in
        names = named_params.keys()
        params = named_params.values()
        # set tangents
        #self.weight_guess = torch.randn_like(self.weight)
        self.weight_guess = self.guess_W(w_next, x_in)/self.num_params
        self.bias_guess = torch.randn_like(self.bias)/self.num_params
        v_params = [
            jvp_in, # x
            self.bias_guess, # bias
            self.weight_guess # weight
        ]
        v_params = tuple(v_params)
        #print(v_params)

        f = partial(
            self.fwd_grad,
            names=names,
            buffers=named_buffers,
        )
        out, jvp_out = fc.jvp(f, (tuple(params),), (v_params,))
        return out, jvp_out # same shape

    def set_grad(self, jvp_val):
        self.bias.grad = jvp_val * self.bias_guess
        self.weight.grad = jvp_val * self.weight_guess

class jvp_out_layer(nn.Module):
    def __init__(self, in_size, out_size, x_param=True):
        super().__init__()
        #f = f_module(in_size, out_size)
        self.x_param = x_param
        if x_param:
            self.x = nn.Parameter(torch.randn(in_size))
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias_guess = None
        self.weight_guess = None
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, dummy_var):  # dummy var since we treat input as a parameter
        if self.x_param:
            return F.linear(self.x, self.weight, self.bias)
        else:
            return F.linear(dummy_var, self.weight, self.bias)

    def fwd_grad(self, params, buffers, names, y, return_loss=True):
        out = fc.functional_call(self, ({k: v for k, v in zip(names, params)}, buffers), (None,))
        if return_loss:
            return self.loss_fn(out, y)
        else:
            return out

    def jvp_forward(self, x_in, jvp_in, y, return_loss=True):
        # get params & buffers
        named_buffers = dict(self.named_buffers())
        named_params = dict(self.named_parameters())
        named_params['x'] = x_in
        names = named_params.keys()
        params = named_params.values()
        # set tangents
        self.weight_guess = torch.randn_like(self.weight)/self.num_params
        self.bias_guess = torch.randn_like(self.bias)/self.num_params
        v_params = [
            jvp_in, # x
            self.bias_guess, # bias
            self.weight_guess # weight
        ]
        v_params = tuple(v_params)
        #print(v_params)

        f = partial(
            self.fwd_grad,
            names=names,
            buffers=named_buffers,
            y=y,
            return_loss=return_loss
        )
        out, jvp_out = fc.jvp(f, (tuple(params),), (v_params,))
        return out, jvp_out # same shape

    def set_grad(self, jvp_val):
        self.bias.grad = jvp_val* self.bias_guess
        self.weight.grad = jvp_val* self.weight_guess

def backprop(tensors):
    m = BackpropLayer(W_i=tensors[0],
                      b_i=tensors[1],
                      W_next=tensors[2],
                      guess=tensors[3])
    grad = m(tensors[4])
    return grad