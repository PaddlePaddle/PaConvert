import paddle
print('#########################case1#########################')
model_parallel_size = paddle.distributed.get_world_size()
print('#########################case2#########################')
wq = paddle.distributed.fleet.meta_parallel.ColumnParallelLinear(in_features
    =args.dim, out_features=args.n_heads * self.head_dim, has_bias=False,
    gather_output=False, weight_attr=paddle.nn.initializer.Constant(0))
print('#########################case3#########################')
wo = paddle.distributed.fleet.meta_parallel.RowParallelLinear(in_features=
    args.n_heads * self.head_dim, out_features=args.dim, has_bias=False,
    input_is_parallel=True, weight_attr=paddle.nn.initializer.Constant(0))
print('#########################case4#########################')
tok_embeddings = paddle.distributed.fleet.meta_parallel.VocabParallelEmbedding(
    num_embeddings=params.vocab_size, embedding_dim=params.dim, weight_attr
    =paddle.nn.initializer.Constant(0))
print('#########################case5#########################')
if not paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None:
    if model_parallel_size is None:
        model_parallel_size = int(os.environ.get('WORLD_SIZE', 1))
print('#########################case6#########################')
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
print('#########################case7#########################')
ckpt_path = checkpoints[paddle.distributed.get_rank()]
