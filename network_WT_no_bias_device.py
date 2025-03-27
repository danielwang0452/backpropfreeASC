
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.func as fc
import math

device = 'mps'
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
            self.output_gradients.insert(0, grad_output[0])  # Store the gradient of the output
        # Register the hook on each linear layer
        for layer in self.net:
            if isinstance(layer, jvp_linear):
                layer.register_backward_hook(save_gradient_hook)

    def forward(self, x, return_loss=False): # will not go through loss fn
        for layer in self.net[:-1]:
            x = layer(x)
        return x

    def jvp_forward(self, x, y):
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
            if self.method == 'weight_perturb':
                layer.weight_guess = torch.randn_like(layer.weight, device=device)
                layer.bias_guess = torch.zeros_like(layer.bias, device=device)
                layer.fwd_method = 'weight_perturb'
            elif self.method == 'act_perturb' or self.method == 'act_perturb-relu':
                layer.s_guess = torch.randn((x.shape[0], layer.weight.shape[0]), device=device) # B x out
                layer.fwd_method = 'act_perturb'
            elif self.method in ['W^T', 'CW^T']:
                if l < len(self.linear_layers) - 1:
                    layer.fwd_method = 'W^T'
                    if self.method == 'CW^T':
                        layer.fwd_method = 'CW^T'
                    # set W_next for all layers except last
                    layer.W_next = self.linear_layers[l + 1].weight.clone()
                if l == len(self.linear_layers) - 1:
                    layer.s_guess = torch.randn((x.shape[0], layer.weight.shape[0]), device=device)  # B x out
                    layer.fwd_method = 'act_perturb'
            elif self.method == 'act_mixing':
                if l == 0 : # first layer has all zero jvp
                    layer.fwd_method = 'act_perturb'
                    layer.s_guess = torch.zeros(x.shape[0], layer.weight.shape[0], device=device)
                elif l == len(self.linear_layers)-1: # last layer has randn guess
                    layer.fwd_method = 'act_mixing'
                    layer.s_guess = torch.randn(x.shape[0], layer.weight.shape[0], device=device)
                else:
                    layer.fwd_method = 'act_mixing'
                    layer.s_guess = 0
            elif self.method == 'layer_downstream':
                layer.fwd_method = 'layer_downstream'
                if l == len(self.linear_layers) - 1:
                    layer.s_guess = torch.randn(x.shape[0], layer.weight.shape[0], device=device)
                    layer.fwd_method = 'act_perturb'
                else:
                    layer.weight_next = self.linear_layers[l+1].weight.clone()

    def set_grad(self, jvp): # set weight & bias grads for each linear layer
        for l, layer in enumerate(self.linear_layers):
            if self.method == 'weight_perturb':
                layer.weight.grad = jvp.sum() * layer.weight_guess
            elif self.method in ['act_perturb',  'act_perturb-relu', 'W^T', 'CW^T', 'layer_downstream']:
                scaled_s_guess = (jvp.unsqueeze(-1) * layer.s_guess)
                layer.weight.grad = (
                    (scaled_s_guess.unsqueeze(2) * layer.x_in.unsqueeze(1))
                ).sum(dim=0)
                layer.bias.grad = scaled_s_guess.sum(dim=0)
                # with torch.enable_grad():
                #    s = F.linear(self.x_in, self.weight, self.bias)
                #    s.backward(gradient=jvp.unsqueeze(-1)*self.s_guess)

            elif self.method == 'act_mixing':
                if l == len(self.linear_layers)-1: # last layer same as act perturb
                    scaled_s_guess = (jvp.unsqueeze(-1) * layer.s_guess)
                    layer.weight.grad = (
                        (scaled_s_guess.unsqueeze(2) * layer.x_in.unsqueeze(1))
                    ).sum(dim=0)
                else: # backprop from guessed activations at start of next layer
                    backprop_grad = jvp.unsqueeze(-1)*self.linear_layers[l+1].x_guess
                    with torch.enable_grad():
                        x_next = F.relu(F.linear(layer.x_in, layer.weight, bias=None))
                        x_next.backward(gradient=backprop_grad)


class jvp_linear(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fwd_method = None
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        self.eye = torch.eye(out_size).to(device)
        #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        #return F.linear(x, self.weight, self.bias)
        return F.linear(x, self.weight, bias=None)
    def act_fwd(self, x): # only track activation jvps for act perturb
        #return F.linear(x, self.weight, self.bias)
        return F.linear(x, self.weight, bias=None)

    def weight_fwd(self, x, bias, weight): # track weight, bias jvps for weight perturb
        return F.linear(x, weight, bias=None)

    def jvp_forward(self, x_in, jvp_in):
        self.x_in = x_in # store x_in for gradient computation
        if self.fwd_method == 'weight_perturb':
            out, jvp_out = fc.jvp(self.weight_fwd, (x_in, self.bias, self.weight), (jvp_in, self.bias_guess, self.weight_guess))
        elif self.fwd_method == 'act_perturb':
            out, jvp = fc.jvp(self.act_fwd, (x_in,), (jvp_in,))
            jvp_out = jvp+self.s_guess
        elif self.fwd_method == 'act_mixing':
            alphas = torch.randn((x_in.shape[0], x_in.shape[0]))
            mixture = alphas @ x_in # B, in
            mixture *= (x_in>0) # relu mask
            # normalise to match the norm of a randn sampled guess? expected norm = sqrt(n)
            self.x_guess = mixture * (torch.tensor(mixture.shape[1], dtype=torch.float32).sqrt()) / (
                (mixture**2).sum(dim=1).sqrt()).unsqueeze(-1)
            out, jvp = fc.jvp(self.act_fwd, (x_in,), (jvp_in + self.x_guess,))
            jvp_out = jvp + self.s_guess
        elif self.fwd_method == 'layer_downstream':
            # backprop then compute jvp for activations
            s_i = nn.Parameter(self(x_in), requires_grad=True)
            with torch.enable_grad():
                s_next = F.relu(F.linear(F.relu(s_i), self.weight_next, bias=None))
                s_next.backward(gradient=torch.randn_like(s_next))
                self.s_guess = s_i.grad * (torch.tensor(s_i.grad.shape[1], dtype=torch.float32).sqrt()) / ((s_i.grad**2).sum(dim=1).sqrt()).unsqueeze(-1)

            out, jvp = fc.jvp(self.act_fwd, (x_in,), (jvp_in,))
            jvp_out = jvp + self.s_guess
            return out, jvp_out
        elif self.fwd_method in ['W^T', 'CW^T']:
            s_next_guess = torch.randn(x_in.shape[0], self.W_next.shape[0], device=device)
            self.mask = ((F.linear(x_in, self.weight, self.bias)) > 0)
            if self.fwd_method == 'W^T':
                self.s_guess = (s_next_guess @ self.W_next) * self.mask # relu mask
                #print((torch.prod(torch.tensor(self.s_guess.shape)).sqrt()), (torch.prod(torch.tensor(x_in.shape)).sqrt()))
                #self.s_guess = self.s_guess*(torch.prod(torch.tensor(self.s_guess.shape)).sqrt())/self.s_guess.norm()
                self.s_guess = self.s_guess #* (torch.tensor(self.s_guess.shape[1], dtype=torch.float32).sqrt()) / ((self.s_guess**2).sum(dim=1).sqrt()).unsqueeze(-1)
            elif self.fwd_method == 'CW^T':
                cov_WT = (self.W_next.T @ self.W_next)  # out, out
                mask = torch.diag_embed(self.mask).to(torch.float32)
                cov_y = (mask @ cov_WT @ mask) + 1e-5*self.eye
                L, Q = torch.linalg.eigh(cov_y)
                self.C = torch.bmm(torch.bmm(Q, torch.diag_embed(L.pow(-1))), Q.permute((0, 2, 1))) # (B, out, out)
                #self.s_guess1 = torch.bmm(self.C_pow, ((s_next_guess @ self.W_next) * self.mask).unsqueeze(-1)).squeeze() # relu mask
                #assert self.C[0] @ cov_y[0] == self.eye
                self.s_guess = torch.bmm(((s_next_guess @ self.W_next) * self.mask).unsqueeze(1), self.C).squeeze()
                #print(self.s_guess2 - self.s_guess1)
            out, jvp = fc.jvp(self.act_fwd, (x_in,), (jvp_in,))
            jvp_out = jvp + self.s_guess
        elif self.fwd_method == 'act_perturb-relu':
            x_next_guess = torch.randn(x_in.shape[0], self.weight.shape[0])
            self.mask = ((F.linear(x_in, self.weight, bias=None)) > 0)
            self.s_guess = x_next_guess * self.mask # relu mask
            #print((torch.prod(torch.tensor(self.s_guess.shape)).sqrt()), (torch.prod(torch.tensor(x_in.shape)).sqrt()))
            #self.s_guess = self.s_guess*(torch.prod(torch.tensor(self.s_guess.shape)).sqrt())/self.s_guess.norm()
            self.s_guess = self.s_guess #* (torch.tensor(self.s_guess.shape[1], dtype=torch.float32).sqrt()) / ((self.s_guess**2).sum(dim=1).sqrt()).unsqueeze(-1)
            #self.s_guess = self.s_guess * (0.001) / self.s_guess.norm()
            out, jvp = fc.jvp(self.act_fwd, (x_in,), (jvp_in,))
            jvp_out = jvp + self.s_guess
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


