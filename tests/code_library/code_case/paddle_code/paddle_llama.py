import paddle
print('#########################case1#########################')
model_parallel_size = paddle.distributed.get_world_size()
print('#########################case2#########################')
wq = paddle.distributed.fleet.meta_parallel.ColumnParallelLinear(in_features
    =args.dim, out_features=args.n_heads * self.head_dim, has_bias=False,
    gather_output=False)
print('#########################case3#########################')
wo = paddle.distributed.fleet.meta_parallel.RowParallelLinear(in_features=
    args.n_heads * self.head_dim, out_features=args.dim, has_bias=False,
    input_is_parallel=True)
print('#########################case4#########################')
tok_embeddings = paddle.distributed.fleet.meta_parallel.VocabParallelEmbedding(
    num_embeddings=params.vocab_size, embedding_dim=params.dim, weight_attr
    =paddle.nn.initializer.Constant(0))
print('#########################case5#########################')
if not paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None:
    if model_parallel_size is None:
        model_parallel_size = int(paddle.distributed.get_world_size())
print('#########################case6#########################')
model_parallel_size_0 = int(min(paddle.distributed.get_world_size(),
    model_parallel_size))
data_parallel_size_0 = int(paddle.distributed.get_world_size() / (
    model_parallel_size_0 * 1))
strategy_0 = paddle.distributed.fleet.DistributedStrategy()
strategy_0.hybrid_configs = dict(dp_degree=data_parallel_size_0, mp_degree=
    model_parallel_size_0, pp_degree=1)
paddle.distributed.fleet.init(is_collective=True, strategy=strategy_0)
print('#########################case7#########################')
ckpt_path = checkpoints[paddle.distributed.get_rank()]
print('#########################case8#########################')
out_0 = paddle.ones(shape=[1, 2, 3])
out_0.stop_gradient = not True
x = out_0
y = paddle.add(x=x, y=paddle.to_tensor(1))
