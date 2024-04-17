import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

import torch
from MultiheadAttention import MultiHeadAttention

class AttentionNetwork(MultiHeadAttention):
    """
    Attention Network
    """
    def __init__(self, 
                 dx: int,
                 dy: int,
                 L: int,
                 num_heads: int,
                 attention_type: str,
                 use_bias: bool = False,
                 dropout_rate: float = 0.0,
                 q_k_v_o_proj_enabled: list = [True, True, True, True],
                 **kwargs):
        """
        Initialize Attention Network. Here, we let qk_embed_size_per_head = dx + dy and vo_embed_size_per_head = dx + dy, meaning that there is no dimension reduction. 
        In particular, the size for q_proj, k_proj, v_proj, and o_proj are:
        q_proj: (num_heads * (dx + dy), dx + dy)
        k_proj: (num_heads * (dx + dy), dx + dy)
        v_proj: (num_heads * (dx + dy), dx + dy)
        o_proj: (dy, num_heads * (dx + dy))

        Args:
            dx (int): input dimension
            dy (int): output dimension
            L (int): sequence length
            num_heads (int): number of heads
            attention_type (str): type of attention
            use_bias (bool, optional): whether to use bias. Defaults to False.
            dropout_rate (float, optional): dropout rate. Defaults to 0.0.
            q_k_v_o_proj_enabled (list, optional): whether to use projection for q, k, v, o. Defaults to [True, True, True, True].
        """
        super().__init__(
            qk_embed_size_per_head = dx + dy, 
            vo_embed_size_per_head = dx + dy, 
            num_heads = num_heads,
            attention_type = attention_type,
            use_bias = use_bias,
            dropout_rate = dropout_rate,
            q_k_v_o_proj_enabled = q_k_v_o_proj_enabled,
            q_dim = dx + dy,
            k_dim = dx + dy,
            v_dim = dx + dy,
            o_dim = dy,
            initialization_method = None,
            **kwargs)
        self.dx = dx
        self.dy = dy
        self.d = dx + dy
        self.L = L
        # self.qk_init = (self.d ** .25) * (self.L ** -.5) * .1
        # self.ov_init_lb = self.qk_init / (self.d ** .25) * .5
        # self.ov_init_ub = self.qk_init / (self.d ** .25) * 1.5
        self.qk_init = self.L ** -.75
        self.ov_init_lb = self.qk_init
        self.ov_init_ub = self.qk_init * 2

        # initialize G matrix indexing each task's position
        self.G = torch.zeros(self.dx, self.dy)
        self.r = self.dx // self.dy
        for i in range(self.dy):
            self.G[i * self.r:(i + 1) * self.r, i] = 1

        # self.initialization()
    
    def get_G(self):
        return self.G
    
    def initialization(self):
        """
        Initialize the parameters of the attention network.
        Here, we initialize all parameters to be diagonal and q_proj = k_proj = I * qk_init, v_proj_y = diag(uniform(ov_init_lb, ov_init_ub)), and o_proj_y = v_proj_y.T.
        """
        # initialize q_proj to be small identity matrix and repeat along the first dimension for num_heads times
        if self.q_k_v_o_proj_enabled[0]:
            self.q_proj.weight.data = torch.eye(self.d).repeat(self._num_heads, 1) * self.qk_init

        # initialize k_proj to be small identity matrix and repeat along the first dimension for num_heads times
        if self.q_k_v_o_proj_enabled[1]:
            self.k_proj.weight.data = torch.eye(self.d).repeat(self._num_heads, 1) * self.qk_init

        # initialize v_proj to be diagonal matrix and repeat the process along the first dimension for num_heads times
        
        assert self._num_heads >= self.dy
        if self.q_k_v_o_proj_enabled[2]:
            self.v_proj.weight.data.fill_(0.0)
            for i in range(self._num_heads):
                tmp = torch.rand(self.d) * (self.ov_init_ub - self.ov_init_lb) + self.ov_init_lb
                # rearrange the elements in tmp such that the last dy elements have the i-th element as the largest element
                tmp_x = tmp[:self.dx]
                tmp_y = tmp[self.dx:]
                max_pos = torch.argmax(tmp_y)
                max_val = tmp_y[max_pos]
                tmp_y[max_pos] = tmp_y[i]
                tmp_y[i] = max_val
                # add a small gap 
                # tmp_y[i] += self.ov_init_lb * .2
                tmp_y[i] += self.ov_init_ub
                tmp_x = torch.zeros_like(tmp_x)

                # simplified initialization
                tmp_y = torch.ones_like(tmp_y) * self.ov_init_lb
                tmp_y[i] = self.ov_init_ub
            
                tmp = torch.cat((tmp_x, tmp_y))


                self.v_proj.weight.data[i * self._vo_embed_size_per_head: (i+1) * self._vo_embed_size_per_head, :] = torch.diag_embed(tmp)
            
        # initialize o_proj to be diagonal matrix the same as v_proj_y and repeat the process along the second dimension for num_heads times
        if self.q_k_v_o_proj_enabled[3]:
            self.o_proj.weight.data.fill_(0.0)
            for i in range(self._num_heads):
                self.o_proj.weight.data[:, i * self._vo_embed_size_per_head + self.dx: (i+1) * self._vo_embed_size_per_head] = self.v_proj.weight.data[i * self._vo_embed_size_per_head + self.dx: (i+1) * self._vo_embed_size_per_head, self.dx:].T
        
    def extract_weights(self):
        """
        Extract the weights of the attention network and combine them for visualization.
        """
        kq_effect_weights, ov_effect_weights, q_proj_weights, k_proj_weights, v_proj_weights, o_proj_weights = self.get_attention_weights()

        # These weights should be diagonal matrices. We extract the diagonal elements for each head and combine them into a matrix with dimension (num_heads, d)
        # kq_effect_weight: (num_heads, k_dim, q_dim)
        kq_effect_weights_diag = kq_effect_weights.view(self._num_heads, self.d, self.d).diagonal(dim1 = 1, dim2 = 2) # (num_heads, d)
        # split into the x and y parts
        kq_effect_weights_diag_x = kq_effect_weights_diag[:, :self.dx]  # (num_heads, dx)
        kq_effect_weights_diag_x_avg = kq_effect_weights_diag_x @ self.G / self.r # (num_heads, dy)
        kq_effect_weights_diag_y = kq_effect_weights_diag[:, self.dx:]  # (num_heads, dy)

        # ov_effect_weight: (num_heads, o_dim, v_dim)
        # split into the x and y parts
        ov_effect_weights_x = ov_effect_weights[:, :, :self.dx]
        ov_effect_weights_y = ov_effect_weights[:, :, self.dx:]
        ov_effect_weights_diag_y = ov_effect_weights_y.diagonal(dim1 = 1, dim2 = 2) # (num_heads, dy)

        return {
            'kq_effect_weights_diag_x_avg': kq_effect_weights_diag_x_avg, # (num_heads, dy)
            'kq_effect_weights_diag_x': kq_effect_weights_diag_x,   # (num_heads, dx)
            'kq_effect_weights_diag_y': kq_effect_weights_diag_y,   # (num_heads, dy)
            'ov_effect_weights_diag_y': ov_effect_weights_diag_y,   # (num_heads, dy)
            'ov_effect_weights_x': ov_effect_weights_x,             # (num_heads, dy, dx)
            'kq_effect_weights': kq_effect_weights,
            'ov_effect_weights': ov_effect_weights,
            'q_proj_weights': q_proj_weights,
            'k_proj_weights': k_proj_weights,
            'v_proj_weights': v_proj_weights,
            'o_proj_weights': o_proj_weights
        }
    
    def visualize_eigenvalues(self, vis, weights_table):
        """
        Visualize the eigenvalues of the attention network.
        """
        # extract the eigenvalues
        kq_effect_weights_diag_x_avg = weights_table['kq_effect_weights_diag_x_avg']
        kq_effect_weights_diag_y = weights_table['kq_effect_weights_diag_y']
        ov_effect_weights_diag_y = weights_table['ov_effect_weights_diag_y']

        # visualize these matrices by heatmap
        vis.heatmap(
            X = kq_effect_weights_diag_x_avg.flip([0]).detach().cpu().numpy(),
            win = 'kq_effect_weights_diag_x_avg',
            # update = 'replace',
            opts = dict(
                title = 'kq_effect_weights_diag_x_avg',
                xlabel = 'x',
                ylabel = 'head',
                colormap = 'Viridis'
            )
        )
        vis.heatmap(
            X = kq_effect_weights_diag_y.flip([0]).detach().cpu().numpy(),
            win = 'kq_effect_weights_diag_y',
            # update = 'replace',
            opts = dict(
                title = 'kq_effect_weights_diag_y',
                xlabel = 'y',
                ylabel = 'head',
                colormap = 'Viridis'
            )
        )
        vis.heatmap(
            X = ov_effect_weights_diag_y.flip([0]).detach().cpu().numpy(),
            win = 'ov_effect_weights_diag_y',
            # update = 'replace',
            opts = dict(
                title = 'ov_effect_weights_diag_y',
                xlabel = 'y',
                ylabel = 'head',
                colormap = 'Viridis'
            )
        )
    
    def visualize_weights(self, vis, weights_table):
        """
        Visualize the weights of the attention network.
        """
        # extract the weights
        q_proj_weights = weights_table['q_proj_weights']
        k_proj_weights = weights_table['k_proj_weights']
        v_proj_weights = weights_table['v_proj_weights']
        o_proj_weights = weights_table['o_proj_weights']

        # visualize these matrices by heatmap by updating the window
        for idx_head in range(self._num_heads):
            vis.heatmap(
                X = q_proj_weights[idx_head].flip([0]).detach().cpu().numpy(),
                win = 'q_proj_weights[{}]'.format(idx_head),
                # update = 'replace',
                opts = dict(
                    title = 'q_proj_weights[{}]'.format(idx_head),
                    colormap = 'Viridis', 
                )
            )
            vis.heatmap(
                X = k_proj_weights[idx_head].flip([0]).detach().cpu().numpy(),
                win = 'k_proj_weights[{}]'.format(idx_head),
                # update = 'replace',
                opts = dict(
                    title = 'k_proj_weights[{}]'.format(idx_head),
                    colormap = 'Viridis', 
                    win = 'k_proj_weights[{}]'.format(idx_head)
                )
            )
            vis.heatmap(
                X = v_proj_weights[idx_head].flip([0]).detach().cpu().numpy(),
                win = 'v_proj_weights[{}]'.format(idx_head),
                # update = 'replace',
                opts = dict(
                    title = 'v_proj_weights[{}]'.format(idx_head),
                    colormap = 'Viridis'
                )
            )
            vis.heatmap(
                X = o_proj_weights[idx_head].flip([0]).detach().cpu().numpy(),
                win = 'o_proj_weights[{}]'.format(idx_head),
                # update = 'replace',
                opts = dict(
                    title = 'o_proj_weights[{}]'.format(idx_head),
                    colormap = 'Viridis'
                )
            )

    # redefine the .to() method to move the attention network to the device
    def to(self, device):
        super().to(device)
        self.G = self.G.to(device)
        return self
        