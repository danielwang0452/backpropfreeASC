import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.func as fc
import math
import numpy as np
import matplotlib.pyplot as plt
import wandb
import json
from newton_schulz.Newton_Schulz import lanczos, get_kth_singular_val, zeropower_via_newtonschulz5
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
        for l, layer in enumerate(self.linear_layers):
            layer.l = l # keep count of layers
        self.output_gradients = []
        self._register_hooks()

    def set_n(self, n): # set the value of n to divide epsilon by for double descent
        for layer in self.linear_layers:
            layer.n = n
        return
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
            elif self.method in ['W^T', 'CW^T', 'CW^T2', 'clip-CW^T2', 'orthogonal_W^T', 'orthogonal_W^T_NS']:
                if l < len(self.linear_layers) - 1:
                    layer.fwd_method = self.method
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
            if layer.fwd_method == 'weight_perturb':
                layer.weight.grad = jvp.sum() * layer.weight_guess
            elif layer.fwd_method in ['act_perturb',  'act_perturb-relu', 'W^T', 'CW^T', 'CW^T2',
                                      'clip-CW^T2, layer_downstream', 'orthogonal_W^T', 'orthogonal_W^T_NS']:

                scaled_s_guess = (jvp.unsqueeze(-1) * layer.s_guess)
                layer.weight.grad = (
                    (scaled_s_guess.unsqueeze(2) * layer.x_in.unsqueeze(1))
                ).sum(dim=0)
                layer.bias.grad = scaled_s_guess.sum(dim=0)
                # with torch.enable_grad():
                #    s = F.linear(self.x_in, self.weight, self.bias)
                #    s.backward(gradient=jvp.unsqueeze(-1)*self.s_guess)

            elif layer.fwd_method == 'act_mixing':
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
            else:
                print('method not found')


class jvp_linear(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fwd_method = None
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        self.eye = torch.eye(out_size).to(device)
        self.count = 0
        self.n = 1.0
        self.k = 10 # used for top-k orthogonal W^T
        #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #nn.init.uniform_(self.bias, -bound, bound)
        self.spectral_norm = None

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

        elif self.fwd_method in ['W^T', 'CW^T','CW^T2', 'clip-CW^T2', 'orthogonal_W^T', 'orthogonal_W^T_NS']:
            s_next_guess = torch.randn(x_in.shape[0], self.W_next.shape[0], device=device)
            s_next_guess# /= torch.tensor(self.n).sqrt()
            self.mask = ((F.linear(x_in, self.weight, self.bias)) > 0)
            #plt.matshow(self.mask.to(torch.float32).clone().cpu().numpy())
            #plt.savefig(f'plots/mask_{self.fwd_method}_layer_{self.l}')
            #plt.close()
            #plt.show()
            if self.fwd_method == 'W^T':
                self.s_guess = (s_next_guess @ self.W_next) * self.mask # relu mask
                #print((torch.prod(torch.tensor(self.s_guess.shape)).sqrt()), (torch.prod(torch.tensor(x_in.shape)).sqrt()))
                #self.s_guess = self.s_guess*(torch.prod(torch.tensor(self.s_guess.shape)).sqrt())/self.s_guess.norm()
                self.s_guess = self.s_guess #* (torch.tensor(self.s_guess.shape[1], dtype=torch.float32).sqrt()) / ((self.s_guess**2).sum(dim=1).sqrt()).unsqueeze(-1)
            elif self.fwd_method == 'CW^T':
                self.sigma = 1e-5
                cov_WT = (self.W_next.T @ self.W_next)  # out, out
                mask = torch.diag_embed(self.mask).to(torch.float32)
                cov_y = (mask @ cov_WT @ mask) + self.sigma*self.eye
                try: # diagonalisation does not always work
                    L, Q = torch.linalg.eigh(cov_y)
                    self.b_eye = self.eye.unsqueeze(0).repeat(x_in.shape[0], 1, 1)
                    self.C = torch.bmm(torch.bmm(Q, torch.diag_embed(L.pow(-0.5))), Q.permute((0, 2, 1)))  # (B, out, out)
                    self.C_inv = torch.bmm(torch.bmm(Q, torch.diag_embed(L.pow(-1.0))), Q.permute((0, 2, 1)))
                    #print(self.C_inv[0])
                    # self.s_guess1 = torch.bmm(self.C_pow, ((s_next_guess @ self.W_next) * self.mask).unsqueeze(-1)).squeeze() # relu mask
                    # assert self.C[0] @ cov_y[0] == self.eye
                    self.s_guess = torch.bmm(((s_next_guess @ self.W_next) * self.mask).unsqueeze(1), self.C).squeeze()
                except: # revert to W^T
                    print("diagonalisation failed")
                    self.s_guess = (s_next_guess @ self.W_next) * self.mask
                '''
                # Sort values in descending order
                sorted_L = np.sort(L.mean(dim=0).clone().cpu().numpy())[::-1]
                rank = torch.linalg.matrix_rank(cov_y).to(torch.float32).mean()
                # Create bar chart
                plt.figure(figsize=(10, 5))
                plt.bar(range(self.weight.shape[0]), sorted_L)
                plt.xlabel(f'Sorted eigenvalues, W^TW mean rank = {rank}')
                plt.ylabel("Eigenvalue")
                plt.title(f"Epoch 0 Layer {self.l+1} Eigenvalues for sigma = {self.sigma}")
                plt.savefig(f'eigenvalue_plots/epoch_0_layer_{self.l+1}_eigenvalues_sigma_{self.sigma}.png')
                plt.close()
                print('saved')
                '''
                #print(((torch.bmm(self.C, torch.bmm(cov_y, self.C)) - self.b_eye)**2).sum())
            elif self.fwd_method in ['CW^T2', 'clip-CW^T2']:
                self.sigma = 1e-5
                cov_WT = (self.W_next.T @ self.W_next)  # out, out
                mask = torch.diag_embed(self.mask).to(torch.float32)
                cov_y = (mask @ cov_WT @ mask) + self.sigma*self.eye
                # compute W' using svd
                W = self.W_next @ mask
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                #print(torch.bmm(torch.bmm(U, torch.diag_embed(S ** 2 + self.sigma).sqrt()), Vh)[0] -
                #      (U[0] * (S[0] ** 2 + self.sigma).sqrt() @ Vh[0]))
                self.W_ = torch.bmm(torch.bmm(U, torch.diag_embed(S ** 2 + self.sigma).sqrt()), Vh)
                #print(torch.bmm(W.permute((0, 2, 1)), W) + self.sigma * self.eye - torch.bmm(self.W_.permute((0, 2, 1)), self.W_))
                # done
                try: # diagonalisation does not always work
                    L, Q = torch.linalg.eigh(cov_y)
                    if self.fwd_method == 'clip-CW^T2':
                        L = torch.clamp(L, min=0.05)
                    self.b_eye = self.eye.unsqueeze(0).repeat(x_in.shape[0], 1, 1)
                    self.C = torch.bmm(torch.bmm(Q, torch.diag_embed(L.pow(-0.5))), Q.permute((0, 2, 1)))  # (B, out, out)
                    #self.C_inv = torch.bmm(torch.bmm(Q, torch.diag_embed(L.p  ow(-1.0))), Q.permute((0, 2, 1)))
                    # self.s_guess1 = torch.bmm(self.C_pow, ((s_next_guess @ self.W_next) * self.mask).unsqueeze(-1)).squeeze() # relu mask
                    # assert self.C[0] @ cov_y[0] == self.eye
                    #print(((s_next_guess[0] @ self.W_next) * self.mask[0] - (s_next_guess[0] @ W[0])).abs().max())
                    #print((s_next_guess[0] @ W[0]) - torch.bmm(s_next_guess.unsqueeze(1), W)[0])
                    #print((((s_next_guess @ self.W_next) * self.mask) - (torch.bmm(s_next_guess.unsqueeze(1), W)).squeeze()).abs().max())
                    self.s_guess = torch.bmm(torch.bmm(s_next_guess.unsqueeze(1), self.W_), self.C).squeeze()
                    #self.s_guess = torch.bmm(((s_next_guess @ self.W_next) * self.mask).unsqueeze(1), self.C).squeeze()
                except: # revert to W^T
                    print("diagonalisation failed")
                    self.s_guess = (s_next_guess @ self.W_next) * self.mask
                    #self.s_guess = self.s_guess * (torch.tensor(self.s_guess.shape[1], dtype=torch.float32).sqrt()) / ((self.s_guess**2).sum(dim=1).sqrt()).unsqueeze(-1)
                #print(torch.tensor(self.s_guess.shape[1], dtype=torch.float32).sqrt(), ((self.s_guess**2).sum(dim=1).sqrt()))
            elif self.fwd_method == 'orthogonal_W^T':
                mask = torch.diag_embed(self.mask).to(torch.float32)
                # compute W' using svd
                W = self.W_next @ mask
                self.rank = int(torch.linalg.matrix_rank(W).to(torch.float32).mean())
                #print(self.rank)
                #print(self.l, self.rank)
                rank=self.k
                rank = 10
                #rank = min(int(torch.linalg.matrix_rank(W).to(torch.float32).mean()), 15)
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                self.U, self.S, self.Vh = U[:, :, :rank], S[:, :rank], Vh[:, :rank, :]
                #self.U, self.S, self.Vh = U[:, :, -10:], S[:, -10:], Vh[:, -10:, :]
                #print(self.U.shape, self.S.shape, self.Vh.shape)
                # set k such that sum k sigma_k^2 = 0.99 sum r sigma_r^2
                '''
                threshold = 0.99 * (self.S.mean(dim=0)**2).sum()
                k = 0
                while (self.S.mean(dim=0)[:k]**2).sum() < threshold:
                    k += 1
                rank = k
                '''
                # only keep the orthonormal columns corresponding to nonzero singular values
                #self.W_ = torch.bmm(torch.bmm(self.U, torch.diag_embed(self.S)), self.Vh)
                self.W_ = torch.bmm(self.U, self.Vh)
                self.s_guess = torch.bmm(s_next_guess.unsqueeze(1), self.W_).squeeze()
            elif self.fwd_method == 'orthogonal_W^T_NS':
                mask = torch.diag_embed(self.mask).to(torch.float32)
                # compute W' using svd
                W = self.W_next @ mask
                self.spectral_norm = lanczos(W[0], 1)[0]
                W_normalised = W/self.spectral_norm
                if self.count % 50 == 0:
                    #U, S, Vh = torch.linalg.svd(W[0], full_matrices=False)
                    top_k_singular_vals = lanczos(W_normalised[0], self.k)
                    #print(top_k_singular_vals[self.k - 1] / self.spectral_norm, self.spectral_norm)
                    key = str(int(10*top_k_singular_vals[self.k-1]))
                    with open('newton_schulz/NS_coeffs.json', 'r') as f:
                        coeffs = json.load(f)
                        self.coefficients = coeffs[key]
                # get normalised singular values
                #spectral_norm = spectral_norm_general(W)
                #size = 512
                #epoch = 300
                #data = {}
                #with open('newton_schulz/normalised_singular_values.json', 'r') as f:
                #    data = json.load(f)
                #with open('newton_schulz/normalised_singular_values.json', 'w') as f:
                #    data[f'size{size}_layer{self.l}_epoch{epoch}'] = S.mean(dim=0).cpu().numpy().tolist()
                #    json.dump(data, f)
                # set NS coefficient
                self.W_ = zeropower_via_newtonschulz5(W_normalised, self.coefficients, 5)
                self.s_guess = torch.bmm(s_next_guess.unsqueeze(1), self.W_).squeeze()
                self.count += 1

            jvp_out = jvp + self.s_guess
        elif self.fwd_method == 'act_perturb-relu':
            x_next_guess = torch.randn(x_in.shape[0], self.weight.shape[0])
            self.mask = ((F.linear(x_in, self.weight, bias=None)) > 0)
            self.s_guess = x_next_guess * self.mask # relu mask
            self.s_guess = self.s_guess #* (torch.tensor(self.s_guess.shape[1], dtype=torch.float32).sqrt()) / ((self.s_guess**2).sum(dim=1).sqrt()).unsqueeze(-1)
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
