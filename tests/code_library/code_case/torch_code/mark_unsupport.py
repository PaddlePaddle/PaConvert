import torch

torch.add

_LOCAL_PROCESS_GROUP = None
_MISSING_LOCAL_PG_ERROR = (
    "Local process group is not yet created! Please use detectron2's `launch()` "
    "to start processes and initialize pytorch process group. If you need to start "
    "processes in other ways, please call comm.create_local_process_group("
    "num_workers_per_machine) after calling torch.distributed.init_process_group()."
)

_MISSING_LOCAL_PG_ERROR = (
    'Local process group is not yet created! Please use detectron2 launch() '
    'to start processes and initialize pytorch process group. If you need to start '
    'processes in other ways, please call comm.create_local_process_group('
    'num_workers_per_machine) after calling torch.distributed.init_process_group().'
)

_MISSING_LOCAL_PG_ERROR = (
    "Local process group is not yet created! Please use detectron2's `launch()` to start processes and initialize pytorch process group. If you need to start processes in other ways, please call comm.create_local_process_group(num_workers_per_machine) after calling torch.distributed.init_process_group()."
)

_MISSING_LOCAL_PG_ERROR = (
    'Local process group is not yet created! Please use detectron2 launch() to start processes and initialize pytorch process group. If you need to start processes in other ways, please call comm.create_local_process_group(num_workers_per_machine) after calling torch.distributed.init_process_group().'
)

_MISSING_LOCAL_PG_ERROR = """
"Local process group is not yet created! Please use detectron2's `launch()` "
"to start processes and initialize pytorch process group. If you need to start "
"processes in other ways, please call comm.create_local_process_group("
"num_workers_per_machine) after calling torch.distributed.init_process_group()."
"""

"""
"Local process group is not yet created! Please use detectron2's `launch()` "
"to start processes and initialize pytorch process group. If you need to start "
"processes in other ways, please call comm.create_local_process_group("
"num_workers_per_machine) after calling torch.distributed.init_process_group()."
"""

_MISSING_LOCAL_PG_ERROR = """
Local process group is not yet created! Please use detectron2's `launch()` 
to start processes and initialize pytorch process group. If you need to start 
processes in other ways, please call comm.create_local_process_group(
num_workers_per_machine) after calling torch.distributed.init_process_group().
"""

"""
Local process group is not yet created! Please use detectron2's `launch()` 
to start processes and initialize pytorch process group. If you need to start 
processes in other ways, please call comm.create_local_process_group(
num_workers_per_machine) after calling torch.distributed.init_process_group().
"""


print("############################################################################")
import paddlenlp
import torch
torch.fake_api(paddlenlp.transformers.BertTokenizer.from_pretrained('bert-base-chinese'), torch.rand(2, 3, 4), paddlenlp.transformers.BertTokenizer.from_pretrained('bert-base-chinese'))
paddlenlp.transformers.BertTokenizer.from_pretrained('bert-base-chinese')


print("#####################################case1##################################")
from torch.utils.cpp_extension import load_inline
source = """
at::Tensor sin_add(at::Tensor x, at::Tensor y) {
    return x.sin() + y.sin();
}
"""
load_inline(
        name='inline_extension',
        cpp_sources=[source],
        functions=['sin_add'])
result = True
print("#########################case2#########################")
print("(Torch")
print("#########################case3#########################")
""" This is an non-standard example
) """
""" (
    This is an non-standard example
) """
""" (
    This is (an non-standard example
) """
print("#########################case4#########################")
print("#########################case5#########################")
""" This is an non-standard example) """
source = """
at::Tensor sin_add(at::Tensor x, at::Tensor y) {
    return x.sin() + y.sin();
}
"""
load_inline(
        name='inline_extension',
        cpp_sources=[source],
        functions=['sin_add'])
result = True
