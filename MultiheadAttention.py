from typing import Optional, Union
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Implementation of multi-head self/cross attention.
    
    In this implementation, the query and key's embedding dimension does not need to be a split of the number of heads, and the attention scores are normalized by the square root of the embedding dimension of the query and key for each head.
    """

    def __init__(
            self,
            qk_embed_size_per_head: int,
            vo_embed_size_per_head: int,
            num_heads: int,
            attention_type: str,
            use_bias: bool = False,
            dropout_rate: float = 0.0,
            q_k_v_o_proj_enabled: list = [True, True, True, True],
            q_dim: int = None,
            k_dim: int = None,
            v_dim: int = None,
            o_dim: int = None,
            initialization_method: str = "small identity",
            **kwargs,
    ):
        """
        Initialize the MultiHeadAttention module.

        Args:
            qk_embed_size_per_head (int): Size of each head for queries and keys.
            v_embed_size (int): Size of each head for values.
            num_heads (int): Number of heads.
            attention_type (str): Type of attention. Can be 'softmax', 'relu' or 'linear'.
            use_bias (bool): Whether to use bias term in projection layers.
            dropout_rate (float): Dropout rate.
            q_k_v_o_proj_enabled (list): List of booleans indicating whether to enable projection layers for queries, keys, values and outputs.
            q_dim (int): Dimension of queries. If None, defaults to qk_embed_size_per_head * num_heads.
            k_dim (int): Dimension of keys. If None, defaults to qk_embed_size_per_head * num_heads.
            v_dim (int): Dimension of values. If None, defaults to v_embed_size * num_heads.
            o_dim (int): Dimension of outputs. If None, defaults to v_embed_size * num_heads.
            initialization_method (str): Initialization method. Can be 'normal', 'small identity' or None.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If qk_embed_size is not divisible by num_heads.
        """
        super().__init__()
        
        # Initialization of dimensions.
        self._qk_embed_size_per_head = qk_embed_size_per_head
        self._vo_embed_size_per_head = vo_embed_size_per_head
        self._num_heads = num_heads
        self.q_k_v_o_proj_enabled = q_k_v_o_proj_enabled


        # Initialization of attention activation.
        if attention_type == 'softmax':
            self.attention_activation = nn.Softmax(dim=-1)
        elif attention_type == 'relu':
            self.attention_activation = nn.ReLU()
        elif attention_type == 'linear':
            self.attention_activation = nn.Identity()
        else:
            raise NotImplementedError(
                f"Attention type {attention_type} is not implemented!"
            )

        # Set the size of for queries, keys and values and outputs.
        if q_dim is None:
            q_dim = self._qk_embed_size_per_head
        if k_dim is None:
            k_dim = self._qk_embed_size_per_head
        if v_dim is None:
            v_dim = self._vo_embed_size_per_head
        if o_dim is None:
            o_dim = self._vo_embed_size_per_head

        self.v_dim = v_dim
        self.k_dim = k_dim
        self.q_dim = q_dim
        self.o_dim = o_dim

        # initialize the q_proj, k_proj, v_proj and o_proj layers for each head
        self.q_proj = nn.Linear(
                in_features=q_dim,
                out_features=self._qk_embed_size_per_head * self._num_heads,
                bias=use_bias,
            ) if q_k_v_o_proj_enabled[0] else nn.Identity()
        
        self.k_proj = nn.Linear(
                in_features=k_dim,
                out_features=self._qk_embed_size_per_head * self._num_heads,
                bias=use_bias,
            ) if q_k_v_o_proj_enabled[1] else nn.Identity()
        
        self.v_proj = nn.Linear(
                in_features=v_dim,
                out_features=self._vo_embed_size_per_head * self._num_heads,
                bias=use_bias,
            ) if q_k_v_o_proj_enabled[2] else nn.Identity()
        
        self.o_proj = nn.Linear(
                in_features=self._vo_embed_size_per_head * self._num_heads,
                out_features=self.o_dim,
                bias=use_bias,
            ) if q_k_v_o_proj_enabled[3] else nn.Identity()

        # Initialization of dropout layer.
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize the weights.
        if initialization_method == "normal":
            if q_k_v_o_proj_enabled[0]:
                self.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            if q_k_v_o_proj_enabled[1]:
                self.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            if q_k_v_o_proj_enabled[2]:
                self.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            if q_k_v_o_proj_enabled[3]:
                self.o_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif initialization_method == "small identity":
            if q_k_v_o_proj_enabled[0]:
                if q_dim == qk_embed_size_per_head:
                    self.q_proj.weight.data = torch.eye(self._qk_embed_size_per_head).repeat(self._num_heads, 1) * 1e-4
                else:
                    raise ValueError(
                        f"q_dim {q_dim} is not equal to qk_embed_size_per_head {qk_embed_size_per_head} for small identity initialization! Please set q_dim to qk_embed_size_per_head."
                    )
            if q_k_v_o_proj_enabled[1]:
                if k_dim == qk_embed_size_per_head:
                    self.k_proj.weight.data = torch.eye(self._qk_embed_size_per_head).repeat(self._num_heads, 1) * 1e-4
                else:
                    raise ValueError(
                        f"k_dim {k_dim} is not equal to qk_embed_size_per_head {qk_embed_size_per_head} for small identity initialization! Please set k_dim to qk_embed_size_per_head."
                    )
            if q_k_v_o_proj_enabled[2]:
                if v_dim == vo_embed_size_per_head:
                    self.v_proj.weight.data = torch.eye(self._vo_embed_size_per_head).repeat(self._num_heads, 1) * 1e-4
                else:
                    raise ValueError(
                        f"v_dim {v_dim} is not equal to vo_embed_size_per_head {vo_embed_size_per_head} for small identity initialization! Please set v_dim to vo_embed_size_per_head."
                    )
            if q_k_v_o_proj_enabled[3]:
                if o_dim == vo_embed_size_per_head:
                    self.o_proj.weight.data = torch.eye(self._vo_embed_size_per_head).repeat(1, self._num_heads) * 1e-4
                else:
                    raise ValueError(
                        f"o_dim {o_dim} is not equal to vo_embed_size_per_head {vo_embed_size_per_head} for small identity initialization! Please set o_dim to vo_embed_size_per_head."
                    )
        elif initialization_method == None:
            pass
        else:
            raise NotImplementedError(
                f"Initialization method {initialization_method} is not implemented!"
            )


    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[Union[torch.Tensor, None]] = None,
    ):
        """
        Apply a forward pass of attention.

        Args:
            query (torch.Tensor): Query tensor 
                of shape [batch_size, query_seq_len, q_dim].
            key (torch.Tensor): Key tensor
                of shape [batch_size, enc_seq_len, k_dim].
            value (torch.Tensor): Value tensor
                of shape [batch_size, enc_seq_len, v_dim].
            mask (Optional[Union[torch.Tensor, None]]): Optional mask tensor 
                of shape [batch_size, 1, x_seq_len, enc_seq_len].

        Returns:
            torch.Tensor: A 3D tensor of shape [batch_size, x_seq_len, qk_embed_size].
        """

        # Linear projections for queries, keys and values.
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to 4D tensors of shape
        # [batch_size, seq_len, num_heads, qkv_size_per_head].
        q = q.view(-1, q.shape[1], self._num_heads, self._qk_embed_size_per_head)
        k = k.view(-1, k.shape[1], self._num_heads, self._qk_embed_size_per_head)
        v = v.view(-1, v.shape[1], self._num_heads, self._vo_embed_size_per_head)

        # Compute attention weights.
        logits = torch.einsum("bnhk,bmhk->bhnm", q, k) * self._qk_embed_size_per_head ** (-0.5)# [batch_size, num_heads, query_seq_len, key_seq_len]
        if mask is not None:
            logits = logits * mask
        weights = self.attention_activation(logits)

        # Apply attention weights dropout.
        weights = self.dropout(weights)

        o = torch.einsum("bhnm,bmhk->bnhk", weights, v) # [batch_size, query_seq_len, num_heads, vo_embed_size_per_head]
        # Reshape to 3D tensor.
        o = torch.reshape(o, (-1, o.shape[1], self._vo_embed_size_per_head * self._num_heads)) # [batch_size, query_seq_len, vo_embed_size]

        # Linear projection for outputs.
        o = self.o_proj(o) # [batch_size, query_seq_len, o_dim]

        return o, weights
    
    def get_attention_weights(self, k_win=None, q_win=None, v_win=None, o_win=None):
        """
        Get the attention weights.
        Here qk_effect_weights = [qk_effect_weight_1, ..., qk_effect_weight_num_heads], where qk_effect_weight_i = q_proj_weights[i].T @ k_proj_weights[i] * qk_embed_size_per_head ** -.5;
        ov_effect_weights = [ov_effect_weight_1, ..., ov_effect_weight_num_heads], where ov_effect_weight_i = o_proj_weights[i] @ v_proj_weights[i];

        Args:
            query (torch.Tensor): Query tensor 
                of shape [batch_size, query_seq_len, q_dim].
            key (torch.Tensor): Key tensor
                of shape [batch_size, enc_seq_len, k_dim].
            mask (Optional[Union[torch.Tensor, None]]): Optional mask tensor 
                of shape [batch_size, 1, x_seq_len, enc_seq_len].

        Returns:
            torch.Tensor: A 4D tensor of shape [batch_size, num_heads, query_seq_len, key_seq_len].
        """

        # Linear projections for queries and keys.
        k_proj_weights = self.k_proj.weight.data if self.q_k_v_o_proj_enabled[1] else torch.eye(self.k_dim).repeat(self._num_heads, 1)
        q_proj_weights = self.q_proj.weight.data if self.q_k_v_o_proj_enabled[0] else torch.eye(self.q_dim).repeat(self._num_heads, 1)

        if k_win is not None:
            k_proj_weights = k_proj_weights * k_win.to(k_proj_weights.device)
        if q_win is not None:
            q_proj_weights = q_proj_weights * q_win.to(q_proj_weights.device)

        # split the weights into num_heads using torch.view method
        k_proj_weights = k_proj_weights.view(self._num_heads, self._qk_embed_size_per_head, self.k_dim) # shape: (num_heads, qk_size_split, k_dim)
        q_proj_weights = q_proj_weights.view(self._num_heads, self._qk_embed_size_per_head, self.q_dim) # shape: (num_heads, qk_size_split, q_dim)

        # compute the attention weights
        kq_effect_weights = torch.einsum("hdk,hdq->hkq", k_proj_weights, q_proj_weights) * self._qk_embed_size_per_head ** -.5 # shape: (num_heads, k_dim, q_dim)

        v_proj_weights = self.v_proj.weight.data if self.q_k_v_o_proj_enabled[2] else torch.eye(self.v_dim).repeat(self._num_heads, 1)
        o_proj_weights = self.o_proj.weight.data if self.q_k_v_o_proj_enabled[3] else torch.eye(self.o_dim).repeat(1, self._num_heads)

        if v_win is not None:
            v_proj_weights = v_proj_weights * v_win.to(v_proj_weights.device)
        if o_win is not None:
            o_proj_weights = o_proj_weights * o_win.to(o_proj_weights.device)

        # split the weights into num_heads
        v_proj_weights = v_proj_weights.view(self._num_heads, self._vo_embed_size_per_head, self.v_dim)  # shape: (num_heads, vo_size_per_head, v_dim)
        o_proj_weights = o_proj_weights.view(self.o_dim, self._num_heads, self._vo_embed_size_per_head).transpose(1, 0)  # shape: (num_heads, o_dim, vo_size_per_head)

        # compute the output weights
        ov_effect_weights = torch.einsum("hod,hdv->hov", o_proj_weights, v_proj_weights) # shape: (num_heads, o_dim, v_dim)

        return kq_effect_weights, ov_effect_weights, q_proj_weights, k_proj_weights, v_proj_weights, o_proj_weights
    

# test code
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    # set the parameters
    batch_size = 1
    query_seq_len = 10
    enc_seq_len = 10
    qk_embed_size_per_head = 2
    vo_embed_size_per_head = 2
    num_heads = 2
    attention_type = 'softmax'
    use_bias = False
    dropout_rate = 0.0
    q_k_v_o_proj_enabled = [True, True, True, True]
    q_dim = 2
    k_dim = 2
    v_dim = 2
    o_dim = 2
    initialization_method = "small identity"

    # create the attention module
    attention = MultiHeadAttention(
        qk_embed_size_per_head=qk_embed_size_per_head,
        vo_embed_size_per_head=vo_embed_size_per_head,
        num_heads=num_heads,
        attention_type=attention_type,
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        q_k_v_o_proj_enabled=q_k_v_o_proj_enabled,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        o_dim=o_dim,
        initialization_method=initialization_method,
    )

    # create the input tensors
    query = torch.randn(batch_size, query_seq_len, q_dim)
    key = torch.randn(batch_size, enc_seq_len, k_dim)
    value = torch.randn(batch_size, enc_seq_len, v_dim)

    # compute the output
    output, weights = attention(query, key, value)

    # get the attention weights
    qk_effect_weights, ov_effect_weights, q_proj_weights, k_proj_weights, v_proj_weights, o_proj_weights = attention.get_attention_weights()
