import sys
sys.path.append('/workspace/workspace/utils')
import paddle_aux
import paddle
import math
from typing import Optional
print('#########################case1#########################')


class Attention(paddle.nn.Layer):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = (args.n_heads if args.n_kv_heads is None else
            args.n_kv_heads)
        model_parallel_size = paddle.distributed.get_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = paddle.distributed.fleet.meta_parallel.ColumnParallelLinear(
            in_features=args.dim, out_features=args.n_heads * self.head_dim,
            has_bias=False, gather_output=False)
        self.wk = paddle.distributed.fleet.meta_parallel.ColumnParallelLinear(
            in_features=args.dim, out_features=self.n_kv_heads * self.
            head_dim, has_bias=False, gather_output=False)
        self.wv = paddle.distributed.fleet.meta_parallel.ColumnParallelLinear(
            in_features=args.dim, out_features=self.n_kv_heads * self.
            head_dim, has_bias=False, gather_output=False)
        self.wo = paddle.distributed.fleet.meta_parallel.RowParallelLinear(
            in_features=args.n_heads * self.head_dim, out_features=args.dim,
            has_bias=False, input_is_parallel=True)
        self.cache_k = paddle.zeros(shape=(args.max_batch_size, args.
            max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.cache_v = paddle.zeros(shape=(args.max_batch_size, args.
            max_seq_len, self.n_local_kv_heads, self.head_dim))

    def forward(self, x: paddle.Tensor, start_pos: int, freqs_cis: paddle.
        Tensor, mask: Optional[paddle.Tensor]):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = tuple(x.shape)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        x = xq
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        xq = x.transpose(perm=perm_0)
        x = keys
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        keys = x.transpose(perm=perm_1)
        x = values
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        values = x.transpose(perm=perm_2)
        x = keys
        perm_3 = list(range(x.ndim))
        perm_3[2] = 3
        perm_3[3] = 2
        scores = paddle.matmul(x=xq, y=x.transpose(perm=perm_3)) / math.sqrt(
            self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = paddle.nn.functional.softmax(x=scores.astype(dtype=
            'float32'), axis=-1).astype(dtype=xq.dtype)
        output = paddle.matmul(x=scores, y=values)
        x = output
        perm_4 = list(range(x.ndim))
        perm_4[1] = 2
        perm_4[2] = 1
        output = x.transpose(perm=perm_4).contiguous()
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)


print('#########################case2#########################')


class Transformer(paddle.nn.Layer):

    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = (paddle.distributed.fleet.meta_parallel.
            VocabParallelEmbedding(num_embeddings=params.vocab_size,
            embedding_dim=params.dim))
        self.layers = paddle.nn.LayerList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = (paddle.distributed.fleet.meta_parallel.
            ColumnParallelLinear(in_features=params.dim, out_features=
            params.vocab_size, has_bias=False, gather_output=True))
        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.
            params.n_heads, self.params.max_seq_len * 2)

    @paddle.no_grad()
    def forward(self, tokens: paddle.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tuple(tokens.shape)
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.place)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = paddle.full(shape=(seqlen, seqlen), fill_value=float('-inf')
                )
            mask = paddle.triu(x=mask, diagonal=1)
            axis_0 = 0 if [paddle.zeros(shape=(seqlen, start_pos)), mask][0
                ].ndim == 1 else 1
            mask = paddle.concat([paddle.zeros(shape=(seqlen, start_pos)),
                mask], axis=axis_0).astype(dtype=h.dtype)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).astype(dtype='float32')
        return output


print('#########################case3#########################')
if not paddle.distributed.is_initialized():
    paddle.distributed.init_parallel_env()
print('#########################case4#########################')
if not paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None:
    if model_parallel_size is None:
        model_parallel_size = int(paddle.distributed.get_world_size())
    world_size_0 = paddle.distributed.get_world_size()
    rank_0 = paddle.distributed.get_rank()
    model_parallel_size_0 = int(min(world_size_0, model_parallel_size))
    data_parallel_size_0 = int(world_size_0 / (model_parallel_size_0 * 1))
    strategy_0 = paddle.distributed.fleet.DistributedStrategy()
    hc_dict_0 = dict()
    hc_dict_0['dp_degree'] = data_parallel_size_0
    hc_dict_0['mp_degree'] = model_parallel_size_0
    hc_dict_0['pp_degree'] = 1
    strategy_0.hybrid_configs = hc_dict_0
    paddle.distributed.fleet.init(is_collective=True, strategy=strategy_0)
print('#########################case5#########################')
ckpt_path = checkpoints[paddle.distributed.get_rank()]
