import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

print("#########################case1#########################")
model_parallel_size_0 = fs_init.get_model_parallel_world_size()    
print("#########################case2#########################")
wq_0 = ColumnParallelLinear(
    dim,
    n_heads * head_dim,
    bias=False,
    gather_output=False,
    init_method=lambda x: x,
)
print("#######################################################") 
wq_1 = ColumnParallelLinear(
    dim,
    n_heads * head_dim,
    gather_output=True,
)
print("#######################################################") 
wq_2 = ColumnParallelLinear(
    dim,
    n_heads * head_dim,
    bias=True,
    gather_output=False,
)
print("#########################case3#########################")
wo_0 = RowParallelLinear(
    n_heads * head_dim,
    dim,
    bias=False,
    input_is_parallel=True,
    init_method=lambda x: x,
)
print("#######################################################") 
wo_1 = RowParallelLinear(
    n_heads * head_dim,
    dim,
    input_is_parallel=False,
)
print("#######################################################") 
wo_2 = RowParallelLinear(
    n_heads * head_dim,
    dim,
    bias=True,
    input_is_parallel=True,
)
print("#########################case4#########################")
tok_embeddings_0 = ParallelEmbedding(
    params.vocab_size, params.dim, init_method=lambda x: x
)
print("#######################################################") 
tok_embeddings_1 = ParallelEmbedding(
    params.vocab_size, params.dim, padding_idx = 0
)
print("#########################case5#########################")
if not model_parallel_is_initialized():
    initialized_flag = False
print("#########################case6#########################")        
initialize_model_parallel(model_parallel_size) 
print("#######################################################") 
initialize_model_parallel(model_parallel_size,pipeline_length)
print("#######################################################") 
initialize_model_parallel(model_parallel_size,model_parallel_backend=model_parallel_backend ,
      pipeline_backend=pipeline_backend,
      ddp_backend=pipeline_backend)
print("#########################case7#########################")
ckpt_path_0 = checkpoints[get_model_parallel_rank()] 
