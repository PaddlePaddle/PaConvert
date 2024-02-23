import torch
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
model_parallel_size = fs_init.get_model_parallel_world_size()     
print("#########################case2#########################")
wq = ColumnParallelLinear(
    args.dim,
    args.n_heads * self.head_dim,
    bias=False,
    gather_output=False,
    init_method=lambda x: x,
)
print("#########################case3#########################")
wo = RowParallelLinear(
    args.n_heads * self.head_dim,
    args.dim,
    bias=False,
    input_is_parallel=True,
    init_method=lambda x: x,
)
print("#########################case4#########################")
tok_embeddings = ParallelEmbedding(
    params.vocab_size, params.dim, init_method=lambda x: x
)
print("#########################case5#########################")
if not model_parallel_is_initialized():
    if model_parallel_size is None:
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
print("#########################case6#########################")        
initialize_model_parallel(model_parallel_size)
print("#########################case7#########################")
ckpt_path = checkpoints[get_model_parallel_rank()]
print("#########################case8#########################")
x = torch.ones(1, 2, 3, requires_grad=True)
y=torch.add(x,1)
