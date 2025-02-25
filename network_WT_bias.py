import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.output_gradients = []
        self._register_hooks()

    # log backprop gradients for testing
    def _register_hooks(self):
        def save_gradient_hook(module, grad_input, grad_output):
            self.output_gradients.insert(0, grad_output[0]) # Store the gradient of the output
        # Register the hook on each linear layer
        for layer in self.net:
            if isinstance(layer, jvp_linear):
                layer.register_backward_hook(save_gradient_hook)

    def forward(self, x, return_loss=False): # will not go through loss fn
        self.output_gradients = []
        for layer in self.net[:-1]:
            x = layer(x)
        return x

    def jvp_forward(self, x, y):
        self.output_gradients = []
        self.prepare_layers(x)
        jvp_in = torch.zeros_like(x)
        for layer in self.net[:-1]:
            x, jvp_in = layer.jvp_forward(x, jvp_in)
        loss_layer = self.net[-1]
        loss_layer.y = y
        loss, jvp_out = loss_layer.jvp_forward(x, jvp_in)
        return loss, jvp_out

    def prepare_layers(self, x):
        # generate randn guesses at appropriate layers for each method
        # set other required attributes e.g layer.W_next for W^T
        # set layer fwd behaviour (track weight jvps or activation jvps)
        for l, layer in enumerate(self.linear_layers):
            if self.method == 'act_perturb':
                layer.s_guess = torch.randn((x.shape[0], layer.weight.shape[0])) # B x out
                layer.fwd_method = 'act_perturb'
            elif self.method == 'W^T':
                if l < len(self.linear_layers)-1:
                    layer.fwd_method = 'W^T'
                    # set W_next for all layers except last
                    layer.W_next = self.linear_layers[l+1].weight.clone()
                if l == len(self.linear_layers)-1:
                    layer.s_guess = torch.randn((x.shape[0], layer.weight.shape[0]))  # B x out
                    layer.fwd_method = 'act_perturb'

    def set_grad(self, jvp): # set weight & bias grads for each linear layer
        for l, layer in enumerate(self.linear_layers):
            if self.method == 'act_perturb' or self.method == 'W^T' or self.method == 'layer_downstream':
                scaled_s_guess = (jvp.unsqueeze(-1) * layer.s_guess)
                layer.weight.grad = (
                    (scaled_s_guess.unsqueeze(2) * layer.x_in.unsqueeze(1))
                ).sum(dim=0)
                layer.bias.grad = scaled_s_guess.sum(dim=0)
                # with torch.enable_grad():
                #    s = F.linear(self.x_in, self.weight, self.bias)
                #    s.backward(gradient=jvp.unsqueeze(-1)*self.s_guess)


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

    def act_fwd(self, x): # only track activation jvps for act perturb
        return F.linear(x, self.weight, self.bias)

    def weight_fwd(self, x, bias, weight): # track weight, bias jvps for weight perturb
        return F.linear(x, weight, bias)

    def jvp_forward(self, x_in, jvp_in):
        self.x_in = x_in # store x_in for gradient computation
        if self.fwd_method == 'weight_perturb':
            out, jvp_out = fc.jvp(self.weight_fwd, (x_in, self.bias, self.weight), (jvp_in, self.bias_guess, self.weight_guess))
        elif self.fwd_method == 'act_perturb':
            out, jvp = fc.jvp(self.act_fwd, (x_in,), (jvp_in,))
            jvp_out = jvp+self.s_guess
        elif self.fwd_method == 'W^T':
            s_next_guess = torch.randn(x_in.shape[0], self.W_next.shape[0])
            self.mask = ((F.linear(x_in, self.weight, self.bias)) > 0)
            self.s_guess = (s_next_guess @ self.W_next) #* self.mask # relu mask
            #print((torch.prod(torch.tensor(self.s_guess.shape)).sqrt()), (torch.prod(torch.tensor(x_in.shape)).sqrt()))
            #self.s_guess = self.s_guess * (torch.tensor(self.s_guess.shape[1], dtype=torch.float32).sqrt()) / ((self.s_guess**2).sum(dim=1).sqrt()).unsqueeze(-1)
            out, jvp = fc.jvp(self.act_fwd, (x_in,), (jvp_in,))
            jvp_out = jvp + self.s_guess
            # used for bias computation

        return out, jvp_out

class jvp_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.y = None

    def forward(self, x, y):
        return self.loss_fn(x, y)

    def func(self, x):
        return self.loss_fn(x, self.y)/x.shape[0]

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
