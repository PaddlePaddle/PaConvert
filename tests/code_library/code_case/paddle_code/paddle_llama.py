import paddle
print('#########################case1#########################')
assert paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None
model_parallel_size_0 = (paddle.distributed.fleet.base.topology.
    _HYBRID_PARALLEL_GROUP._mp_degree)
print('#########################case2#########################')
wq_0 = paddle.distributed.fleet.meta_parallel.ColumnParallelLinear(in_features
    =dim, out_features=n_heads * head_dim, has_bias=False, gather_output=False)
print('#######################################################')
wq_1 = paddle.distributed.fleet.meta_parallel.ColumnParallelLinear(in_features
    =dim, out_features=n_heads * head_dim, gather_output=True, has_bias=True)
print('#######################################################')
wq_2 = paddle.distributed.fleet.meta_parallel.ColumnParallelLinear(in_features
    =dim, out_features=n_heads * head_dim, has_bias=True, gather_output=False)
print('#########################case3#########################')
wo_0 = paddle.distributed.fleet.meta_parallel.RowParallelLinear(in_features
    =n_heads * head_dim, out_features=dim, has_bias=False,
    input_is_parallel=True)
print('#######################################################')
wo_1 = paddle.distributed.fleet.meta_parallel.RowParallelLinear(in_features
    =n_heads * head_dim, out_features=dim, input_is_parallel=False,
    has_bias=True)
print('#######################################################')
wo_2 = paddle.distributed.fleet.meta_parallel.RowParallelLinear(in_features
    =n_heads * head_dim, out_features=dim, has_bias=True, input_is_parallel
    =True)
print('#########################case4#########################')
tok_embeddings_0 = (paddle.distributed.fleet.meta_parallel.
    VocabParallelEmbedding(num_embeddings=params.vocab_size, embedding_dim=
    params.dim, weight_attr=paddle.nn.initializer.Constant(0)))
print('#######################################################')
>>>>>>tok_embeddings_1 = fairscale.nn.model_parallel.layers.ParallelEmbedding(params
    .vocab_size, params.dim, padding_idx=0)
print('#########################case5#########################')
if not paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None:
    initialized_flag = False
print('#########################case6#########################')
model_parallel_size_0 = int(min(paddle.distributed.get_world_size(),
    model_parallel_size))
data_parallel_size_0 = int(paddle.distributed.get_world_size() / (
    model_parallel_size_0 * 1))
strategy_0 = paddle.distributed.fleet.DistributedStrategy()
strategy_0.hybrid_configs = dict(dp_degree=data_parallel_size_0, mp_degree=
    model_parallel_size_0, pp_degree=1)
paddle.distributed.fleet.init(is_collective=True, strategy=strategy_0)
print('#######################################################')
model_parallel_size_1 = int(min(paddle.distributed.get_world_size(),
    model_parallel_size))
data_parallel_size_1 = int(paddle.distributed.get_world_size() / (
    model_parallel_size_1 * pipeline_length))
strategy_1 = paddle.distributed.fleet.DistributedStrategy()
strategy_1.hybrid_configs = dict(dp_degree=data_parallel_size_1, mp_degree=
    model_parallel_size_1, pp_degree=pipeline_length)
paddle.distributed.fleet.init(is_collective=True, strategy=strategy_1)
print('#######################################################')
model_parallel_size_2 = int(min(paddle.distributed.get_world_size(),
    model_parallel_size))
data_parallel_size_2 = int(paddle.distributed.get_world_size() / (
    model_parallel_size_2 * 1))
strategy_2 = paddle.distributed.fleet.DistributedStrategy()
strategy_2.hybrid_configs = dict(dp_degree=data_parallel_size_2, mp_degree=
    model_parallel_size_2, pp_degree=1)
paddle.distributed.fleet.init(is_collective=True, strategy=strategy_2)
print('#########################case7#########################')
assert paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None
ckpt_path_0 = checkpoints[paddle.distributed.fleet.base.topology.
    _HYBRID_PARALLEL_GROUP.get_model_parallel_rank()]
