# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch

# torch.BFloat16Storage
result = torch.BFloat16Storage

# torch.BoolStorage
result = torch.BoolStorage

# torch.ByteStorage
result = torch.ByteStorage

# torch.CharStorage
result = torch.CharStorage

# torch.ComplexDoubleStorage
result = torch.ComplexDoubleStorage

# torch.ComplexFloatStorage
result = torch.ComplexFloatStorage

# torch.DoubleStorage
result = torch.DoubleStorage

# torch.FloatStorage
result = torch.FloatStorage

# torch.HalfStorage
result = torch.HalfStorage

# torch.IntStorage
result = torch.IntStorage

# torch.LongStorage
result = torch.LongStorage

# torch.QInt32Storage
result = torch.QInt32Storage

# torch.QInt8Storage
result = torch.QInt8Storage

# torch.QUInt2x4Storage
result = torch.QUInt2x4Storage

# torch.QUInt4x2Storage
result = torch.QUInt4x2Storage

# torch.QUInt8Storage
result = torch.QUInt8Storage

# torch.ShortStorage
result = torch.ShortStorage

# torch.SymBool
result = torch.SymBool

# torch.SymFloat
result = torch.SymFloat

# torch.SymInt
result = torch.SymInt

# torch.Tag
result = torch.Tag

# torch.Tensor.align_as
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
result = x.align_as(y)

# torch.Tensor.align_to
x = torch.randn(2, 3, 4)
result = x.align_to("N", "C", "H")

# torch.Tensor.as_subclass
x = torch.randn(2, 3, 4)
result = x.as_subclass(torch.Tensor)

# torch.Tensor.ccol_indices
x = torch.randn(2, 3, 4)
result = x.ccol_indices()

# torch.Tensor.chalf
x = torch.randn(2, 3, 4)
result = x.chalf()

# torch.Tensor.conj_physical_
x = torch.randn(2, 3, 4)
result = x.conj_physical_()

# torch.Tensor.dequantize
x = torch.randn(2, 3, 4)
result = x.dequantize()

# torch.Tensor.geqrf
x = torch.randn(2, 3, 4)
result = x.geqrf()

# torch.Tensor.index_copy
x = torch.randn(2, 3, 4)
dim = 1
index = torch.tensor([0, 1], dtype=torch.int64)
src = torch.randn(2, 3, 4)
result = x.index_copy(dim, index, src)

# torch.Tensor.index_reduce
x = torch.randn(2, 3, 4)
dim = 1
index = torch.tensor([0, 1], dtype=torch.int64)
src = torch.randn(2, 3, 4)
result = x.index_reduce(dim, index, src, reduce="sum")

# torch.Tensor.index_reduce_
x = torch.randn(2, 3, 4)
dim = 1
index = torch.tensor([0, 1], dtype=torch.int64)
src = torch.randn(2, 3, 4)
result = x.index_reduce_(dim, index, src, reduce="sum")

# torch.Tensor.is_conj
x = torch.randn(2, 3, 4)
result = x.is_conj()

# torch.Tensor.is_meta
x = torch.randn(2, 3, 4)
result = x.is_meta

# torch.Tensor.is_quantized
x = torch.randn(2, 3, 4)
result = x.is_quantized

# torch.Tensor.is_set_to
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
result = x.is_set_to(y)

# torch.Tensor.is_shared
x = torch.randn(2, 3, 4)
result = x.is_shared()

# torch.Tensor.map_
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
result = x.map_(y, lambda a, b: a + b)

# torch.Tensor.module_load
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
result = x.module_load(y, assign=False)

# torch.Tensor.names
x = torch.randn(2, 3, 4)
result = x.names

# torch.Tensor.nextafter_
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
result = x.nextafter_(y)

# torch.Tensor.put_
x = torch.randn(2, 3, 4)
index = torch.tensor([0, 1], dtype=torch.int64)
values = torch.tensor([1.0, 2.0])
result = x.put_(index, values)

# torch.Tensor.q_per_channel_axis
x = torch.randn(2, 3, 4)
result = x.q_per_channel_axis()

# torch.Tensor.q_per_channel_scales
x = torch.randn(2, 3, 4)
result = x.q_per_channel_scales()

# torch.Tensor.q_per_channel_zero_points
x = torch.randn(2, 3, 4)
result = x.q_per_channel_zero_points()

# torch.Tensor.q_scale
x = torch.randn(2, 3, 4)
result = x.q_scale()

# torch.Tensor.q_zero_point
x = torch.randn(2, 3, 4)
result = x.q_zero_point()

# torch.Tensor.qscheme
x = torch.randn(2, 3, 4)
result = x.qscheme()

# torch.Tensor.record_stream
x = torch.randn(2, 3, 4)
result = x.record_stream(torch.cuda.current_stream() if torch.cuda.is_available() else None)

# torch.Tensor.refine_names
x = torch.randn(2, 3, 4)
result = x.refine_names("N", "C", "H")

# torch.Tensor.register_post_accumulate_grad_hook
x = torch.randn(2, 3, 4)
result = x.register_post_accumulate_grad_hook(lambda grad: grad)

# torch.Tensor.rename
x = torch.randn(2, 3, 4)
result = x.rename("N", "C", "H")

# torch.Tensor.rename_
x = torch.randn(2, 3, 4)
result = x.rename_("N", "C", "H")

# torch.Tensor.resolve_conj
x = torch.randn(2, 3, 4)
result = x.resolve_conj()

# torch.Tensor.resolve_neg
x = torch.randn(2, 3, 4)
result = x.resolve_neg()

# torch.Tensor.retains_grad
x = torch.randn(2, 3, 4)
result = x.retains_grad

# torch.Tensor.row_indices
x = torch.randn(2, 3, 4)
result = x.row_indices()

# torch.Tensor.scatter_reduce_
x = torch.randn(2, 3, 4)
dim = 1
src = torch.randn(2, 3, 4)
scatter_index = torch.tensor([[[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]], dtype=torch.int64)
result = x.scatter_reduce_(dim, scatter_index, src, reduce="sum")

# torch.Tensor.sgn_
x = torch.randn(2, 3, 4)
result = x.sgn_()

# torch.Tensor.share_memory_
x = torch.randn(2, 3, 4)
result = x.share_memory_()

# torch.Tensor.sign_
x = torch.randn(2, 3, 4)
result = x.sign_()

# torch.Tensor.smm
x = torch.randn(2, 3, 4)
result = x.smm()

# torch.Tensor.sparse_resize_
x = torch.randn(2, 3, 4)
result = x.sparse_resize_((2, 3, 4), 0, 0)

# torch.Tensor.sparse_resize_and_clear_
x = torch.randn(2, 3, 4)
result = x.sparse_resize_and_clear_((2, 3, 4), 0, 0)

# torch.Tensor.sspaddmm
x = torch.randn(2, 3, 4)
result = x.sspaddmm()

# torch.Tensor.storage
x = torch.randn(2, 3, 4)
result = x.storage()

# torch.Tensor.storage_offset
x = torch.randn(2, 3, 4)
result = x.storage_offset()

# torch.Tensor.storage_type
x = torch.randn(2, 3, 4)
result = x.storage_type()

# torch.Tensor.sum_to_size
x = torch.randn(2, 3, 4)
result = x.sum_to_size(2, 3, 4)

# torch.Tensor.to_mkldnn
x = torch.randn(2, 3, 4)
result = x.to_mkldnn()

# torch.Tensor.to_sparse_bsc
x = torch.randn(2, 3, 4)
result = x.to_sparse_bsc()

# torch.Tensor.to_sparse_bsr
x = torch.randn(2, 3, 4)
result = x.to_sparse_bsr()

# torch.Tensor.to_sparse_csc
x = torch.randn(2, 3, 4)
result = x.to_sparse_csc()

# torch.Tensor.untyped_storage
x = torch.randn(2, 3, 4)
result = x.untyped_storage()

# torch.TypedStorage
result = torch.TypedStorage

# torch.UntypedStorage
result = torch.UntypedStorage

# torch.__config__.parallel_info
result = torch.__config__.parallel_info()

# torch.__config__.show
result = torch.__config__.show()

# torch.__future__.get_overwrite_module_params_on_conversion
result = torch.__future__.get_overwrite_module_params_on_conversion()

# torch.__future__.get_swap_module_params_on_conversion
result = torch.__future__.get_swap_module_params_on_conversion()

# torch.__future__.set_overwrite_module_params_on_conversion
result = torch.__future__.set_overwrite_module_params_on_conversion(True)

# torch.__future__.set_swap_module_params_on_conversion
result = torch.__future__.set_swap_module_params_on_conversion(True)

# torch._logging.set_logs
result = torch._logging.set_logs(True)

# torch.are_deterministic_algorithms_enabled
result = torch.are_deterministic_algorithms_enabled()

# torch.autograd.Function.jvp
result = torch.autograd.Function.jvp()

# torch.autograd.Function.vmap
result = torch.autograd.Function.vmap()

# torch.autograd.detect_anomaly
result = torch.autograd.detect_anomaly()

# torch.autograd.forward_ad.UnpackedDualTensor
result = torch.autograd.forward_ad.UnpackedDualTensor

# torch.autograd.forward_ad.dual_level
result = torch.autograd.forward_ad.dual_level()

# torch.autograd.forward_ad.enter_dual_level
result = torch.autograd.forward_ad.enter_dual_level()

# torch.autograd.forward_ad.exit_dual_level
result = torch.autograd.forward_ad.exit_dual_level()

# torch.autograd.forward_ad.make_dual
x = torch.randn(2, 3, 4)
result = torch.autograd.forward_ad.make_dual(x)

# torch.autograd.forward_ad.unpack_dual
x = torch.randn(2, 3, 4)
result = torch.autograd.forward_ad.unpack_dual(x)

# torch.autograd.function.BackwardCFunction
result = torch.autograd.function.BackwardCFunction

# torch.autograd.function.FunctionCtx.mark_dirty
result = torch.autograd.function.FunctionCtx.mark_dirty()

# torch.autograd.function.InplaceFunction
result = torch.autograd.function.InplaceFunction

# torch.autograd.function.NestedIOFunction
result = torch.autograd.function.NestedIOFunction

# torch.autograd.function.once_differentiable
result = torch.autograd.function.once_differentiable(lambda *args, **kwargs: None)

# torch.autograd.functional.hvp
x = torch.randn(2, 3, 4)
result = torch.autograd.functional.hvp(x)

# torch.autograd.functional.vhp
x = torch.randn(2, 3, 4)
result = torch.autograd.functional.vhp(x)

# torch.autograd.grad_mode.inference_mode
result = torch.autograd.grad_mode.inference_mode()

# torch.autograd.grad_mode.set_multithreading_enabled
result = torch.autograd.grad_mode.set_multithreading_enabled(True)

# torch.autograd.gradcheck.GradcheckError
result = torch.autograd.gradcheck.GradcheckError

# torch.autograd.gradcheck.gradcheck
result = torch.autograd.gradcheck.gradcheck()

# torch.autograd.gradcheck.gradgradcheck
result = torch.autograd.gradcheck.gradgradcheck()

# torch.autograd.graph.GradientEdge
result = torch.autograd.graph.GradientEdge

# torch.autograd.graph.Node.metadata
result = torch.autograd.graph.Node.metadata

# torch.autograd.graph.Node.name
result = torch.autograd.graph.Node.name

# torch.autograd.graph.Node.next_functions
result = torch.autograd.graph.Node.next_functions

# torch.autograd.graph.Node.register_hook
result = torch.autograd.graph.Node.register_hook(lambda *args, **kwargs: None)

# torch.autograd.graph.Node.register_prehook
result = torch.autograd.graph.Node.register_prehook(lambda *args, **kwargs: None)

# torch.autograd.graph.allow_mutation_on_saved_tensors
result = torch.autograd.graph.allow_mutation_on_saved_tensors()

# torch.autograd.graph.disable_saved_tensors_hooks
result = torch.autograd.graph.disable_saved_tensors_hooks()

# torch.autograd.graph.get_gradient_edge
result = torch.autograd.graph.get_gradient_edge()

# torch.autograd.graph.increment_version
x = torch.randn(2, 3, 4)
result = torch.autograd.graph.increment_version(x)

# torch.autograd.graph.register_multi_grad_hook
result = torch.autograd.graph.register_multi_grad_hook()

# torch.autograd.graph.save_on_cpu
result = torch.autograd.graph.save_on_cpu()

# torch.autograd.profiler.EnforceUnique
result = torch.autograd.profiler.EnforceUnique

# torch.autograd.profiler.KinetoStepTracker
result = torch.autograd.profiler.KinetoStepTracker

# torch.autograd.profiler.emit_itt
result = torch.autograd.profiler.emit_itt()

# torch.autograd.profiler.emit_nvtx
result = torch.autograd.profiler.emit_nvtx()

# torch.autograd.profiler.load_nvprof
result = torch.autograd.profiler.load_nvprof("tmp.pt")

# torch.autograd.profiler.parse_nvprof_trace
result = torch.autograd.profiler.parse_nvprof_trace()

# torch.autograd.profiler.profile.key_averages
result = torch.autograd.profiler.profile.key_averages()

# torch.autograd.profiler.profile.self_cpu_time_total
result = torch.autograd.profiler.profile.self_cpu_time_total

# torch.autograd.profiler.profile.total_average
result = torch.autograd.profiler.profile.total_average()

# torch.autograd.profiler.profile
result = torch.autograd.profiler.profile()

# torch.autograd.profiler.record_function
result = torch.autograd.profiler.record_function("tag")

# torch.autograd.profiler_util.Interval
result = torch.autograd.profiler_util.Interval

# torch.autograd.profiler_util.Kernel
result = torch.autograd.profiler_util.Kernel

# torch.autograd.profiler_util.MemRecordsAcc
result = torch.autograd.profiler_util.MemRecordsAcc

# torch.autograd.profiler_util.StringTable
result = torch.autograd.profiler_util.StringTable

# torch.autograd.set_detect_anomaly
result = torch.autograd.set_detect_anomaly(True)

# torch.backends.cpu.get_cpu_capability
result = torch.backends.cpu.get_cpu_capability()

# torch.backends.cuda.SDPAParams
result = torch.backends.cuda.SDPAParams

# torch.backends.cuda.can_use_efficient_attention
result = torch.backends.cuda.can_use_efficient_attention()

# torch.backends.cuda.can_use_flash_attention
result = torch.backends.cuda.can_use_flash_attention()

# torch.backends.cuda.cudnn_sdp_enabled
result = torch.backends.cuda.cudnn_sdp_enabled()

# torch.backends.cuda.enable_cudnn_sdp
result = torch.backends.cuda.enable_cudnn_sdp(True)

# torch.backends.cuda.enable_flash_sdp
result = torch.backends.cuda.enable_flash_sdp(True)

# torch.backends.cuda.enable_math_sdp
result = torch.backends.cuda.enable_math_sdp(True)

# torch.backends.cuda.enable_mem_efficient_sdp
result = torch.backends.cuda.enable_mem_efficient_sdp(True)

# torch.backends.cuda.flash_sdp_enabled
result = torch.backends.cuda.flash_sdp_enabled()

# torch.backends.cuda.math_sdp_enabled
result = torch.backends.cuda.math_sdp_enabled()

# torch.backends.cuda.mem_efficient_sdp_enabled
result = torch.backends.cuda.mem_efficient_sdp_enabled()

# torch.backends.cuda.preferred_linalg_library
x = torch.randn(2, 3, 4)
result = torch.backends.cuda.preferred_linalg_library(x)

# torch.backends.cuda.sdp_kernel
x = torch.randn(2, 3, 4)
result = torch.backends.cuda.sdp_kernel(x)

# torch.backends.mha.get_fastpath_enabled
result = torch.backends.mha.get_fastpath_enabled()

# torch.backends.mha.set_fastpath_enabled
result = torch.backends.mha.set_fastpath_enabled(True)

# torch.backends.mkl.is_available
result = torch.backends.mkl.is_available()

# torch.backends.mkl.verbose
result = torch.backends.mkl.verbose()

# torch.backends.mkldnn.is_available
result = torch.backends.mkldnn.is_available()

# torch.backends.mkldnn.verbose
result = torch.backends.mkldnn.verbose()

# torch.backends.mps.is_available
result = torch.backends.mps.is_available()

# torch.backends.mps.is_built
result = torch.backends.mps.is_built()

# torch.backends.nnpack.flags
result = torch.backends.nnpack.flags()

# torch.backends.nnpack.is_available
result = torch.backends.nnpack.is_available()

# torch.backends.nnpack.set_flags
result = torch.backends.nnpack.set_flags(True)

# torch.backends.openmp.is_available
result = torch.backends.openmp.is_available()

# torch.backends.opt_einsum.get_opt_einsum
result = torch.backends.opt_einsum.get_opt_einsum()

# torch.backends.opt_einsum.is_available
result = torch.backends.opt_einsum.is_available()

# torch.bartlett_window
x = torch.randn(2, 3, 4)
result = torch.bartlett_window(x)

# torch.compile
result = torch.compile()

# torch.compiled_with_cxx11_abi
result = torch.compiled_with_cxx11_abi()

# torch.cond
x = torch.randn(2, 3, 4)
result = torch.cond(x)

# torch.cpu.StreamContext
result = torch.cpu.StreamContext

# torch.cpu.Stream
result = torch.cpu.Stream

# torch.cpu.current_stream
result = torch.cpu.current_stream()

# torch.cpu.device_count
result = torch.cpu.device_count()

# torch.cpu.is_available
result = torch.cpu.is_available()

# torch.cpu.stream
result = torch.cpu.stream()

# torch.cpu.synchronize
result = torch.cpu.synchronize()

# torch.cuda.CUDAGraph
result = torch.cuda.CUDAGraph

# torch.cuda.CUDAPluggableAllocator
result = torch.cuda.CUDAPluggableAllocator

# torch.cuda.ExternalStream
result = torch.cuda.ExternalStream

# torch.cuda.OutOfMemoryError
result = torch.cuda.OutOfMemoryError

# torch.cuda.amp.custom_bwd
result = torch.cuda.amp.custom_bwd()

# torch.cuda.amp.custom_fwd
result = torch.cuda.amp.custom_fwd()

# torch.cuda.caching_allocator_alloc
result = torch.cuda.caching_allocator_alloc()

# torch.cuda.caching_allocator_delete
result = torch.cuda.caching_allocator_delete()

# torch.cuda.can_device_access_peer
result = torch.cuda.can_device_access_peer()

# torch.cuda.change_current_allocator
result = torch.cuda.change_current_allocator()

# torch.cuda.clock_rate
result = torch.cuda.clock_rate()

# torch.cuda.comm.broadcast_coalesced
x = torch.randn(2, 3, 4)
result = torch.cuda.comm.broadcast_coalesced(x)

# torch.cuda.comm.gather
x = torch.randn(2, 3, 4)
result = torch.cuda.comm.gather(x)

# torch.cuda.comm.reduce_add
result = torch.cuda.comm.reduce_add()

# torch.cuda.comm.scatter
x = torch.randn(2, 3, 4)
result = torch.cuda.comm.scatter(x)

# torch.cuda.current_blas_handle
result = torch.cuda.current_blas_handle()

# torch.cuda.default_stream
result = torch.cuda.default_stream()

# torch.cuda.device_of
x = torch.randn(2, 3, 4)
result = torch.cuda.device_of(x)

# torch.cuda.get_allocator_backend
result = torch.cuda.get_allocator_backend()

# torch.cuda.get_arch_list
result = torch.cuda.get_arch_list()

# torch.cuda.get_gencode_flags
result = torch.cuda.get_gencode_flags()

# torch.cuda.get_sync_debug_mode
result = torch.cuda.get_sync_debug_mode()

# torch.cuda.graph
x = torch.randn(2, 3, 4)
result = torch.cuda.graph(x)

# torch.cuda.graph_pool_handle
x = torch.randn(2, 3, 4)
result = torch.cuda.graph_pool_handle(x)

# torch.cuda.init
result = torch.cuda.init()

# torch.cuda.jiterator._create_jit_fn
result = torch.cuda.jiterator._create_jit_fn()

# torch.cuda.jiterator._create_multi_output_jit_fn
result = torch.cuda.jiterator._create_multi_output_jit_fn()

# torch.cuda.list_gpu_processes
result = torch.cuda.list_gpu_processes()

# torch.cuda.make_graphed_callables
result = torch.cuda.make_graphed_callables()

# torch.cuda.max_memory_cached
result = torch.cuda.max_memory_cached()

# torch.cuda.memory._dump_snapshot
result = torch.cuda.memory._dump_snapshot()

# torch.cuda.memory._record_memory_history
result = torch.cuda.memory._record_memory_history()

# torch.cuda.memory._snapshot
result = torch.cuda.memory._snapshot()

# torch.cuda.memory_cached
result = torch.cuda.memory_cached()

# torch.cuda.memory_snapshot
result = torch.cuda.memory_snapshot()

# torch.cuda.memory_stats
result = torch.cuda.memory_stats()

# torch.cuda.memory_summary
result = torch.cuda.memory_summary()

# torch.cuda.memory_usage
result = torch.cuda.memory_usage()

# torch.cuda.nvtx.mark
result = torch.cuda.nvtx.mark("tag")

# torch.cuda.power_draw
result = torch.cuda.power_draw()

# torch.cuda.seed
result = torch.cuda.seed()

# torch.cuda.seed_all
result = torch.cuda.seed_all()

# torch.cuda.set_sync_debug_mode
result = torch.cuda.set_sync_debug_mode(True)

# torch.cuda.temperature
result = torch.cuda.temperature()

# torch.cuda.utilization
result = torch.cuda.utilization()

# torch.dequantize
x = torch.randn(2, 3, 4)
result = torch.dequantize(x)

# torch.distributed.DistBackendError
result = torch.distributed.DistBackendError

# torch.distributed.DistError
result = torch.distributed.DistError

# torch.distributed.DistNetworkError
result = torch.distributed.DistNetworkError

# torch.distributed.DistStoreError
result = torch.distributed.DistStoreError

# torch.distributed.FileStore
result = torch.distributed.FileStore

# torch.distributed.GradBucket.buffer
result = torch.distributed.GradBucket.buffer()

# torch.distributed.GradBucket.gradients
result = torch.distributed.GradBucket.gradients()

# torch.distributed.GradBucket.index
result = torch.distributed.GradBucket.index()

# torch.distributed.GradBucket.is_last
result = torch.distributed.GradBucket.is_last()

# torch.distributed.GradBucket.parameters
result = torch.distributed.GradBucket.parameters()

# torch.distributed.GradBucket.set_buffer
result = torch.distributed.GradBucket.set_buffer(True)

# torch.distributed.GradBucket
result = torch.distributed.GradBucket

# torch.distributed.HashStore
result = torch.distributed.HashStore

# torch.distributed.PrefixStore
result = torch.distributed.PrefixStore

# torch.distributed.Store.add
result = torch.distributed.Store.add()

# torch.distributed.Store.compare_set
result = torch.distributed.Store.compare_set()

# torch.distributed.Store.delete_key
result = torch.distributed.Store.delete_key()

# torch.distributed.Store.get
result = torch.distributed.Store.get()

# torch.distributed.Store.num_keys
result = torch.distributed.Store.num_keys()

# torch.distributed.Store.set
result = torch.distributed.Store.set()

# torch.distributed.Store.set_timeout
result = torch.distributed.Store.set_timeout(True)

# torch.distributed.Store.wait
result = torch.distributed.Store.wait()

# torch.distributed.Store
result = torch.distributed.Store

# torch.distributed.TCPStore
result = torch.distributed.TCPStore

# torch.distributed.Work
result = torch.distributed.Work

# torch.distributed.algorithms.JoinHook
result = torch.distributed.algorithms.JoinHook

# torch.distributed.algorithms.Join
result = torch.distributed.algorithms.Join

# torch.distributed.algorithms.Joinable
result = torch.distributed.algorithms.Joinable

# torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook
x = torch.randn(2, 3, 4)
result = torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook(x)

# torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook
x = torch.randn(2, 3, 4)
result = torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook(x)

# torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook
x = torch.randn(2, 3, 4)
result = torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook(x)

# torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper
x = torch.randn(2, 3, 4)
result = torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper(x)

# torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
x = torch.randn(2, 3, 4)
result = torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook(x)

# torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper
x = torch.randn(2, 3, 4)
result = torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper(x)

# torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState
result = torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState

# torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook
x = torch.randn(2, 3, 4)
result = torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook(x)

# torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook
x = torch.randn(2, 3, 4)
result = torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook(x)

# torch.distributed.autograd.backward
x = torch.randn(2, 3, 4)
result = torch.distributed.autograd.backward(x)

# torch.distributed.autograd.context
result = torch.distributed.autograd.context()

# torch.distributed.autograd.get_gradients
result = torch.distributed.autograd.get_gradients()

# torch.distributed.breakpoint
result = torch.distributed.breakpoint()

# torch.distributed.checkpoint.DefaultLoadPlanner
result = torch.distributed.checkpoint.DefaultLoadPlanner

# torch.distributed.checkpoint.DefaultSavePlanner
result = torch.distributed.checkpoint.DefaultSavePlanner

# torch.distributed.checkpoint.LoadPlan
result = torch.distributed.checkpoint.LoadPlan

# torch.distributed.checkpoint.LoadPlanner
result = torch.distributed.checkpoint.LoadPlanner

# torch.distributed.checkpoint.ReadItem
result = torch.distributed.checkpoint.ReadItem

# torch.distributed.checkpoint.SavePlan
result = torch.distributed.checkpoint.SavePlan

# torch.distributed.checkpoint.SavePlanner
result = torch.distributed.checkpoint.SavePlanner

# torch.distributed.checkpoint.StorageReader
result = torch.distributed.checkpoint.StorageReader

# torch.distributed.checkpoint.StorageWriter
result = torch.distributed.checkpoint.StorageWriter

# torch.distributed.checkpoint.filesystem.FileSystemReader
result = torch.distributed.checkpoint.filesystem.FileSystemReader

# torch.distributed.checkpoint.filesystem.FileSystemWriter
result = torch.distributed.checkpoint.filesystem.FileSystemWriter

# torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader
result = torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader

# torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner
result = torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner

# torch.distributed.checkpoint.format_utils.dcp_to_torch_save
x = torch.randn(2, 3, 4)
result = torch.distributed.checkpoint.format_utils.dcp_to_torch_save(x)

# torch.distributed.checkpoint.format_utils.torch_save_to_dcp
x = torch.randn(2, 3, 4)
result = torch.distributed.checkpoint.format_utils.torch_save_to_dcp(x)

# torch.distributed.checkpoint.fsspec.FsspecReader
result = torch.distributed.checkpoint.fsspec.FsspecReader

# torch.distributed.checkpoint.fsspec.FsspecWriter
result = torch.distributed.checkpoint.fsspec.FsspecWriter

# torch.distributed.checkpoint.planner.WriteItem
result = torch.distributed.checkpoint.planner.WriteItem

# torch.distributed.checkpoint.state_dict.StateDictOptions
result = torch.distributed.checkpoint.state_dict.StateDictOptions

# torch.distributed.checkpoint.state_dict.get_model_state_dict
result = torch.distributed.checkpoint.state_dict.get_model_state_dict()

# torch.distributed.checkpoint.state_dict.get_optimizer_state_dict
result = torch.distributed.checkpoint.state_dict.get_optimizer_state_dict()

# torch.distributed.checkpoint.state_dict.get_state_dict
result = torch.distributed.checkpoint.state_dict.get_state_dict()

# torch.distributed.checkpoint.state_dict.set_model_state_dict
result = torch.distributed.checkpoint.state_dict.set_model_state_dict(True)

# torch.distributed.checkpoint.state_dict.set_optimizer_state_dict
result = torch.distributed.checkpoint.state_dict.set_optimizer_state_dict(True)

# torch.distributed.checkpoint.state_dict.set_state_dict
result = torch.distributed.checkpoint.state_dict.set_state_dict(True)

# torch.distributed.checkpoint.state_dict_loader.load
result = torch.distributed.checkpoint.state_dict_loader.load("tmp.pt")

# torch.distributed.checkpoint.state_dict_loader.load_state_dict
result = torch.distributed.checkpoint.state_dict_loader.load_state_dict("tmp.pt")

# torch.distributed.checkpoint.state_dict_saver.async_save
x = torch.randn(2, 3, 4)
result = torch.distributed.checkpoint.state_dict_saver.async_save(x, "tmp.pt")

# torch.distributed.checkpoint.state_dict_saver.save
x = torch.randn(2, 3, 4)
result = torch.distributed.checkpoint.state_dict_saver.save(x, "tmp.pt")

# torch.distributed.checkpoint.state_dict_saver.save_state_dict
x = torch.randn(2, 3, 4)
result = torch.distributed.checkpoint.state_dict_saver.save_state_dict(x, "tmp.pt")

# torch.distributed.checkpoint.stateful.Stateful
result = torch.distributed.checkpoint.stateful.Stateful

# torch.distributed.device_mesh.DeviceMesh
result = torch.distributed.device_mesh.DeviceMesh

# torch.distributed.device_mesh.init_device_mesh
result = torch.distributed.device_mesh.init_device_mesh()

# torch.distributed.fsdp.BackwardPrefetch
result = torch.distributed.fsdp.BackwardPrefetch

# torch.distributed.fsdp.CPUOffload
result = torch.distributed.fsdp.CPUOffload

# torch.distributed.fsdp.FullOptimStateDictConfig
result = torch.distributed.fsdp.FullOptimStateDictConfig

# torch.distributed.fsdp.FullStateDictConfig
result = torch.distributed.fsdp.FullStateDictConfig

# torch.distributed.fsdp.FullyShardedDataParallel
result = torch.distributed.fsdp.FullyShardedDataParallel

# torch.distributed.fsdp.LocalOptimStateDictConfig
result = torch.distributed.fsdp.LocalOptimStateDictConfig

# torch.distributed.fsdp.LocalStateDictConfig
result = torch.distributed.fsdp.LocalStateDictConfig

# torch.distributed.fsdp.MixedPrecision
result = torch.distributed.fsdp.MixedPrecision

# torch.distributed.fsdp.OptimStateDictConfig
result = torch.distributed.fsdp.OptimStateDictConfig

# torch.distributed.fsdp.ShardedOptimStateDictConfig
result = torch.distributed.fsdp.ShardedOptimStateDictConfig

# torch.distributed.fsdp.ShardedStateDictConfig
result = torch.distributed.fsdp.ShardedStateDictConfig

# torch.distributed.fsdp.ShardingStrategy
result = torch.distributed.fsdp.ShardingStrategy

# torch.distributed.fsdp.StateDictConfig
result = torch.distributed.fsdp.StateDictConfig

# torch.distributed.fsdp.StateDictSettings
result = torch.distributed.fsdp.StateDictSettings

# torch.distributed.gather_object
x = torch.randn(2, 3, 4)
result = torch.distributed.gather_object(x)

# torch.distributed.get_global_rank
result = torch.distributed.get_global_rank()

# torch.distributed.get_group_rank
result = torch.distributed.get_group_rank()

# torch.distributed.is_gloo_available
result = torch.distributed.is_gloo_available()

# torch.distributed.is_mpi_available
result = torch.distributed.is_mpi_available()

# torch.distributed.is_torchelastic_launched
result = torch.distributed.is_torchelastic_launched()

# torch.distributed.nn.api.remote_module.RemoteModule
result = torch.distributed.nn.api.remote_module.RemoteModule

# torch.distributed.optim.PostLocalSGDOptimizer
result = torch.distributed.optim.PostLocalSGDOptimizer

# torch.distributed.optim.ZeroRedundancyOptimizer
result = torch.distributed.optim.ZeroRedundancyOptimizer

# torch.distributed.pipeline.sync.Pipe
result = torch.distributed.pipeline.sync.Pipe

# torch.distributed.pipeline.sync.skip.skippable.pop
result = torch.distributed.pipeline.sync.skip.skippable.pop()

# torch.distributed.pipeline.sync.skip.skippable.skippable
result = torch.distributed.pipeline.sync.skip.skippable.skippable(lambda *args, **kwargs: None)

# torch.distributed.pipeline.sync.skip.skippable.stash
x = torch.randn(2, 3, 4)
result = torch.distributed.pipeline.sync.skip.skippable.stash(x)

# torch.distributed.pipeline.sync.skip.skippable.verify_skippables
result = torch.distributed.pipeline.sync.skip.skippable.verify_skippables()

# torch.distributed.reduce_op
result = torch.distributed.reduce_op()

# torch.distributed.rpc.BackendType
result = torch.distributed.rpc.BackendType

# torch.distributed.rpc.PyRRef
result = torch.distributed.rpc.PyRRef

# torch.distributed.rpc.RpcBackendOptions
result = torch.distributed.rpc.RpcBackendOptions

# torch.distributed.rpc.TensorPipeRpcBackendOptions
result = torch.distributed.rpc.TensorPipeRpcBackendOptions

# torch.distributed.rpc.WorkerInfo
result = torch.distributed.rpc.WorkerInfo

# torch.distributed.rpc.functions.async_execution
result = torch.distributed.rpc.functions.async_execution()

# torch.distributed.tensor.parallel.ColwiseParallel
result = torch.distributed.tensor.parallel.ColwiseParallel

# torch.distributed.tensor.parallel.PrepareModuleInput
result = torch.distributed.tensor.parallel.PrepareModuleInput

# torch.distributed.tensor.parallel.PrepareModuleOutput
result = torch.distributed.tensor.parallel.PrepareModuleOutput

# torch.distributed.tensor.parallel.RowwiseParallel
result = torch.distributed.tensor.parallel.RowwiseParallel

# torch.distributed.tensor.parallel.SequenceParallel
result = torch.distributed.tensor.parallel.SequenceParallel

# torch.distributed.tensor.parallel.loss_parallel
result = torch.distributed.tensor.parallel.loss_parallel()

# torch.distributed.tensor.parallel.parallelize_module
result = torch.distributed.tensor.parallel.parallelize_module()

# torch.distributions.constraint_registry.ConstraintRegistry
result = torch.distributions.constraint_registry.ConstraintRegistry

# torch.distributions.fishersnedecor.FisherSnedecor
result = torch.distributions.fishersnedecor.FisherSnedecor

# torch.distributions.half_cauchy.HalfCauchy
result = torch.distributions.half_cauchy.HalfCauchy

# torch.distributions.half_normal.HalfNormal
result = torch.distributions.half_normal.HalfNormal

# torch.distributions.inverse_gamma.InverseGamma
result = torch.distributions.inverse_gamma.InverseGamma

# torch.distributions.kumaraswamy.Kumaraswamy
result = torch.distributions.kumaraswamy.Kumaraswamy

# torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal
result = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal

# torch.distributions.mixture_same_family.MixtureSameFamily
result = torch.distributions.mixture_same_family.MixtureSameFamily

# torch.distributions.negative_binomial.NegativeBinomial
result = torch.distributions.negative_binomial.NegativeBinomial

# torch.distributions.pareto.Pareto
result = torch.distributions.pareto.Pareto

# torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli
result = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli

# torch.distributions.relaxed_bernoulli.RelaxedBernoulli
result = torch.distributions.relaxed_bernoulli.RelaxedBernoulli

# torch.distributions.relaxed_categorical.RelaxedOneHotCategorical
result = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical

# torch.distributions.transforms.CorrCholeskyTransform
result = torch.distributions.transforms.CorrCholeskyTransform

# torch.distributions.transforms.LowerCholeskyTransform
result = torch.distributions.transforms.LowerCholeskyTransform

# torch.distributions.von_mises.VonMises
result = torch.distributions.von_mises.VonMises

# torch.distributions.weibull.Weibull
result = torch.distributions.weibull.Weibull

# torch.distributions.wishart.Wishart
result = torch.distributions.wishart.Wishart

# torch.empty_strided
result = torch.empty_strided()

# torch.export.ExportBackwardSignature
result = torch.export.ExportBackwardSignature

# torch.export.ExportGraphSignature
result = torch.export.ExportGraphSignature

# torch.export.ExportedProgram
result = torch.export.ExportedProgram

# torch.export.ModuleCallEntry
result = torch.export.ModuleCallEntry

# torch.export.ModuleCallSignature
result = torch.export.ModuleCallSignature

# torch.export.dims
result = torch.export.dims()

# torch.export.dynamic_shapes.Dim
result = torch.export.dynamic_shapes.Dim

# torch.export.dynamic_shapes.dynamic_dim
result = torch.export.dynamic_shapes.dynamic_dim()

# torch.export.export
result = torch.export.export()

# torch.export.graph_signature.CustomObjArgument
result = torch.export.graph_signature.CustomObjArgument

# torch.export.graph_signature.ExportGraphSignature
result = torch.export.graph_signature.ExportGraphSignature

# torch.export.graph_signature.InputKind
result = torch.export.graph_signature.InputKind

# torch.export.graph_signature.InputSpec
result = torch.export.graph_signature.InputSpec

# torch.export.graph_signature.OutputKind
result = torch.export.graph_signature.OutputKind

# torch.export.graph_signature.OutputSpec
result = torch.export.graph_signature.OutputSpec

# torch.export.load
result = torch.export.load("tmp.pt")

# torch.export.register_dataclass
result = torch.export.register_dataclass()

# torch.export.save
x = torch.randn(2, 3, 4)
result = torch.export.save(x, "tmp.pt")

# torch.export.unflatten.FlatArgsAdapter
result = torch.export.unflatten.FlatArgsAdapter

# torch.export.unflatten.InterpreterModule
result = torch.export.unflatten.InterpreterModule

# torch.export.unflatten.unflatten
result = torch.export.unflatten.unflatten()

# torch.fake_quantize_per_channel_affine
result = torch.fake_quantize_per_channel_affine()

# torch.fake_quantize_per_tensor_affine
result = torch.fake_quantize_per_tensor_affine()

# torch.from_file
result = torch.from_file()

# torch.futures.Future
result = torch.futures.Future

# torch.futures.collect_all
result = torch.futures.collect_all()

# torch.futures.wait_all
result = torch.futures.wait_all()

# torch.fx.GraphModule
result = torch.fx.GraphModule

# torch.fx.Graph
result = torch.fx.Graph

# torch.fx.Interpreter
result = torch.fx.Interpreter

# torch.fx.Node
result = torch.fx.Node

# torch.fx.Proxy
result = torch.fx.Proxy

# torch.fx.Tracer
result = torch.fx.Tracer

# torch.fx.Transformer
result = torch.fx.Transformer

# torch.fx.experimental.symbolic_shapes.DimConstraints
result = torch.fx.experimental.symbolic_shapes.DimConstraints

# torch.fx.experimental.symbolic_shapes.DimDynamic
result = torch.fx.experimental.symbolic_shapes.DimDynamic

# torch.fx.experimental.symbolic_shapes.EqualityConstraint
result = torch.fx.experimental.symbolic_shapes.EqualityConstraint

# torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint
result = torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint

# torch.fx.experimental.symbolic_shapes.ShapeEnv
result = torch.fx.experimental.symbolic_shapes.ShapeEnv

# torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext
result = torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext

# torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext
result = torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext

# torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint
result = torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint

# torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext
result = torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext

# torch.fx.experimental.symbolic_shapes.SymbolicContext
result = torch.fx.experimental.symbolic_shapes.SymbolicContext

# torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr
result = torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr()

# torch.fx.experimental.symbolic_shapes.constrain_range
result = torch.fx.experimental.symbolic_shapes.constrain_range()

# torch.fx.experimental.symbolic_shapes.constrain_unify
result = torch.fx.experimental.symbolic_shapes.constrain_unify()

# torch.fx.experimental.symbolic_shapes.definitely_false
result = torch.fx.experimental.symbolic_shapes.definitely_false()

# torch.fx.experimental.symbolic_shapes.definitely_true
result = torch.fx.experimental.symbolic_shapes.definitely_true()

# torch.fx.experimental.symbolic_shapes.guard_size_oblivious
result = torch.fx.experimental.symbolic_shapes.guard_size_oblivious()

# torch.fx.experimental.symbolic_shapes.has_free_symbols
result = torch.fx.experimental.symbolic_shapes.has_free_symbols()

# torch.fx.experimental.symbolic_shapes.hint_int
result = torch.fx.experimental.symbolic_shapes.hint_int()

# torch.fx.experimental.symbolic_shapes.is_concrete_bool
result = torch.fx.experimental.symbolic_shapes.is_concrete_bool()

# torch.fx.experimental.symbolic_shapes.is_concrete_int
result = torch.fx.experimental.symbolic_shapes.is_concrete_int()

# torch.fx.experimental.symbolic_shapes.parallel_and
result = torch.fx.experimental.symbolic_shapes.parallel_and()

# torch.fx.experimental.symbolic_shapes.parallel_or
result = torch.fx.experimental.symbolic_shapes.parallel_or()

# torch.fx.experimental.symbolic_shapes.statically_known_true
result = torch.fx.experimental.symbolic_shapes.statically_known_true()

# torch.fx.experimental.symbolic_shapes.sym_eq
result = torch.fx.experimental.symbolic_shapes.sym_eq()

# torch.fx.replace_pattern
result = torch.fx.replace_pattern()

# torch.fx.symbolic_trace
result = torch.fx.symbolic_trace()

# torch.fx.wrap
result = torch.fx.wrap()

# torch.geqrf
result = torch.geqrf()

# torch.get_deterministic_debug_mode
result = torch.get_deterministic_debug_mode()

# torch.get_float32_matmul_precision
result = torch.get_float32_matmul_precision()

# torch.gradient
result = torch.gradient()

# torch.hspmm
result = torch.hspmm()

# torch.hub.get_dir
result = torch.hub.get_dir()

# torch.hub.set_dir
result = torch.hub.set_dir(True)

# torch.index_reduce
result = torch.index_reduce()

# torch.is_conj
result = torch.is_conj()

# torch.is_deterministic_algorithms_warn_only_enabled
result = torch.is_deterministic_algorithms_warn_only_enabled()

# torch.is_inference_mode_enabled
result = torch.is_inference_mode_enabled()

# torch.is_storage
result = torch.is_storage()

# torch.is_warn_always_enabled
result = torch.is_warn_always_enabled()

# torch.jit.Attribute
result = torch.jit.Attribute

# torch.jit.ScriptFunction
result = torch.jit.ScriptFunction

# torch.jit.ScriptModule
result = torch.jit.ScriptModule

# torch.jit.annotate
result = torch.jit.annotate()

# torch.jit.enable_onednn_fusion
result = torch.jit.enable_onednn_fusion(True)

# torch.jit.export
result = torch.jit.export()

# torch.jit.fork
result = torch.jit.fork()

# torch.jit.freeze
result = torch.jit.freeze()

# torch.jit.interface
result = torch.jit.interface()

# torch.jit.isinstance
result = torch.jit.isinstance()

# torch.jit.onednn_fusion_enabled
result = torch.jit.onednn_fusion_enabled()

# torch.jit.optimize_for_inference
result = torch.jit.optimize_for_inference()

# torch.jit.script_if_tracing
result = torch.jit.script_if_tracing()

# torch.jit.set_fusion_strategy
result = torch.jit.set_fusion_strategy(True)

# torch.jit.strict_fusion
result = torch.jit.strict_fusion()

# torch.jit.trace
result = torch.jit.trace()

# torch.jit.trace_module
result = torch.jit.trace_module()

# torch.jit.unused
result = torch.jit.unused()

# torch.jit.wait
result = torch.jit.wait()

# torch.kaiser_window
result = torch.kaiser_window()

# torch.layout
result = torch.layout()

# torch.library.Library
result = torch.library.Library

# torch.library.define
result = torch.library.define()

# torch.library.fallthrough_kernel
result = torch.library.fallthrough_kernel()

# torch.library.get_ctx
result = torch.library.get_ctx()

# torch.library.impl
result = torch.library.impl()

# torch.library.impl_abstract
result = torch.library.impl_abstract()

# torch.linalg.ldl_factor
result = torch.linalg.ldl_factor()

# torch.linalg.ldl_factor_ex
result = torch.linalg.ldl_factor_ex()

# torch.linalg.ldl_solve
result = torch.linalg.ldl_solve()

# torch.linalg.solve_ex
result = torch.linalg.solve_ex()

# torch.linalg.tensorinv
result = torch.linalg.tensorinv()

# torch.linalg.tensorsolve
result = torch.linalg.tensorsolve()

# torch.lobpcg
result = torch.lobpcg()

# torch.memory_format
result = torch.memory_format()

# torch.monitor.Aggregation
result = torch.monitor.Aggregation

# torch.monitor.EventHandlerHandle
result = torch.monitor.EventHandlerHandle

# torch.monitor.Event
result = torch.monitor.Event

# torch.monitor.Stat
result = torch.monitor.Stat

# torch.monitor.TensorboardEventHandler
result = torch.monitor.TensorboardEventHandler

# torch.monitor.data_value_t
result = torch.monitor.data_value_t()

# torch.monitor.log_event
result = torch.monitor.log_event()

# torch.monitor.register_event_handler
result = torch.monitor.register_event_handler()

# torch.monitor.unregister_event_handler
result = torch.monitor.unregister_event_handler()

# torch.mps.current_allocated_memory
result = torch.mps.current_allocated_memory()

# torch.mps.driver_allocated_memory
result = torch.mps.driver_allocated_memory()

# torch.mps.empty_cache
result = torch.mps.empty_cache()

# torch.mps.event.Event
result = torch.mps.event.Event

# torch.mps.get_rng_state
result = torch.mps.get_rng_state()

# torch.mps.manual_seed
result = torch.mps.manual_seed(2024)

# torch.mps.profiler.profile
result = torch.mps.profiler.profile()

# torch.mps.profiler.start
result = torch.mps.profiler.start()

# torch.mps.profiler.stop
result = torch.mps.profiler.stop()

# torch.mps.seed
result = torch.mps.seed()

# torch.mps.set_per_process_memory_fraction
result = torch.mps.set_per_process_memory_fraction(True)

# torch.mps.set_rng_state
result = torch.mps.set_rng_state(True)

# torch.mps.synchronize
result = torch.mps.synchronize()

# torch.nested.as_nested_tensor
result = torch.nested.as_nested_tensor()

# torch.nested.nested_tensor
result = torch.nested.nested_tensor()

# torch.nested.to_padded_tensor
result = torch.nested.to_padded_tensor()

# torch.nn.EmbeddingBag
result = torch.nn.EmbeddingBag

# torch.nn.LPPool3d
result = torch.nn.LPPool3d

# torch.nn.attention.bias
result = torch.nn.attention.bias()

# torch.nn.functional.embedding_bag
result = torch.nn.functional.embedding_bag()

# torch.nn.functional.lp_pool3d
result = torch.nn.functional.lp_pool3d()

# torch.nn.init.sparse_
result = torch.nn.init.sparse_()

# torch.nn.modules.lazy.LazyModuleMixin
result = torch.nn.modules.lazy.LazyModuleMixin

# torch.nn.modules.module.register_module_backward_hook
result = torch.nn.modules.module.register_module_backward_hook()

# torch.nn.modules.module.register_module_buffer_registration_hook
result = torch.nn.modules.module.register_module_buffer_registration_hook()

# torch.nn.modules.module.register_module_full_backward_hook
result = torch.nn.modules.module.register_module_full_backward_hook()

# torch.nn.modules.module.register_module_full_backward_pre_hook
result = torch.nn.modules.module.register_module_full_backward_pre_hook()

# torch.nn.modules.module.register_module_module_registration_hook
result = torch.nn.modules.module.register_module_module_registration_hook()

# torch.nn.modules.module.register_module_parameter_registration_hook
result = torch.nn.modules.module.register_module_parameter_registration_hook()

# torch.nn.parameter.UninitializedBuffer
result = torch.nn.parameter.UninitializedBuffer

# torch.nn.utils.convert_conv2d_weight_memory_format
result = torch.nn.utils.convert_conv2d_weight_memory_format()

# torch.nn.utils.convert_conv3d_weight_memory_format
result = torch.nn.utils.convert_conv3d_weight_memory_format()

# torch.nn.utils.fuse_conv_bn_eval
result = torch.nn.utils.fuse_conv_bn_eval()

# torch.nn.utils.fuse_conv_bn_weights
result = torch.nn.utils.fuse_conv_bn_weights()

# torch.nn.utils.fuse_linear_bn_eval
result = torch.nn.utils.fuse_linear_bn_eval()

# torch.nn.utils.fuse_linear_bn_weights
result = torch.nn.utils.fuse_linear_bn_weights()

# torch.nn.utils.parametrizations.orthogonal
result = torch.nn.utils.parametrizations.orthogonal()

# torch.nn.utils.parametrize.ParametrizationList
result = torch.nn.utils.parametrize.ParametrizationList

# torch.nn.utils.parametrize.cached
result = torch.nn.utils.parametrize.cached()

# torch.nn.utils.parametrize.is_parametrized
result = torch.nn.utils.parametrize.is_parametrized()

# torch.nn.utils.parametrize.register_parametrization
result = torch.nn.utils.parametrize.register_parametrization()

# torch.nn.utils.parametrize.remove_parametrizations
result = torch.nn.utils.parametrize.remove_parametrizations()

# torch.nn.utils.prune.BasePruningMethod
result = torch.nn.utils.prune.BasePruningMethod

# torch.nn.utils.prune.CustomFromMask
result = torch.nn.utils.prune.CustomFromMask

# torch.nn.utils.prune.Identity
result = torch.nn.utils.prune.Identity

# torch.nn.utils.prune.L1Unstructured
result = torch.nn.utils.prune.L1Unstructured

# torch.nn.utils.prune.LnStructured
result = torch.nn.utils.prune.LnStructured

# torch.nn.utils.prune.PruningContainer
result = torch.nn.utils.prune.PruningContainer

# torch.nn.utils.prune.RandomStructured
result = torch.nn.utils.prune.RandomStructured

# torch.nn.utils.prune.RandomUnstructured
result = torch.nn.utils.prune.RandomUnstructured

# torch.nn.utils.prune.custom_from_mask
result = torch.nn.utils.prune.custom_from_mask()

# torch.nn.utils.prune.global_unstructured
result = torch.nn.utils.prune.global_unstructured()

# torch.nn.utils.prune.identity
result = torch.nn.utils.prune.identity()

# torch.nn.utils.prune.is_pruned
result = torch.nn.utils.prune.is_pruned()

# torch.nn.utils.prune.l1_unstructured
result = torch.nn.utils.prune.l1_unstructured()

# torch.nn.utils.prune.ln_structured
result = torch.nn.utils.prune.ln_structured()

# torch.nn.utils.prune.random_structured
result = torch.nn.utils.prune.random_structured()

# torch.nn.utils.prune.random_unstructured
result = torch.nn.utils.prune.random_unstructured()

# torch.nn.utils.prune.remove
result = torch.nn.utils.prune.remove()

# torch.nn.utils.remove_spectral_norm
result = torch.nn.utils.remove_spectral_norm()

# torch.nn.utils.rnn.PackedSequence
result = torch.nn.utils.rnn.PackedSequence

# torch.nn.utils.rnn.pack_padded_sequence
result = torch.nn.utils.rnn.pack_padded_sequence()

# torch.nn.utils.rnn.pack_sequence
result = torch.nn.utils.rnn.pack_sequence()

# torch.nn.utils.rnn.pad_packed_sequence
result = torch.nn.utils.rnn.pad_packed_sequence()

# torch.nn.utils.rnn.unpack_sequence
result = torch.nn.utils.rnn.unpack_sequence()

# torch.nn.utils.skip_init
result = torch.nn.utils.skip_init()

# torch.nn.utils.stateless.functional_call
result = torch.nn.utils.stateless.functional_call()

# torch.optim.SparseAdam
result = torch.optim.SparseAdam

# torch.optim.lr_scheduler.ChainedScheduler
result = torch.optim.lr_scheduler.ChainedScheduler

# torch.optim.lr_scheduler.PolynomialLR
result = torch.optim.lr_scheduler.PolynomialLR

# torch.optim.lr_scheduler.SequentialLR
result = torch.optim.lr_scheduler.SequentialLR

# torch.overrides.get_ignored_functions
result = torch.overrides.get_ignored_functions()

# torch.overrides.get_overridable_functions
result = torch.overrides.get_overridable_functions()

# torch.overrides.get_testing_overrides
result = torch.overrides.get_testing_overrides()

# torch.overrides.handle_torch_function
result = torch.overrides.handle_torch_function()

# torch.overrides.has_torch_function
result = torch.overrides.has_torch_function()

# torch.overrides.is_tensor_like
result = torch.overrides.is_tensor_like()

# torch.overrides.is_tensor_method_or_property
result = torch.overrides.is_tensor_method_or_property()

# torch.overrides.resolve_name
result = torch.overrides.resolve_name()

# torch.overrides.wrap_torch_function
result = torch.overrides.wrap_torch_function()

# torch.package.Directory
result = torch.package.Directory

# torch.package.EmptyMatchError
result = torch.package.EmptyMatchError

# torch.package.PackageExporter
result = torch.package.PackageExporter

# torch.package.PackageImporter
result = torch.package.PackageImporter

# torch.package.PackagingError
result = torch.package.PackagingError

# torch.profiler.ProfilerAction
result = torch.profiler.ProfilerAction

# torch.profiler.ProfilerActivity
result = torch.profiler.ProfilerActivity

# torch.profiler._KinetoProfile
result = torch.profiler._KinetoProfile()

# torch.profiler.itt.is_available
result = torch.profiler.itt.is_available()

# torch.profiler.itt.mark
result = torch.profiler.itt.mark("tag")

# torch.profiler.itt.range_pop
result = torch.profiler.itt.range_pop()

# torch.profiler.itt.range_push
result = torch.profiler.itt.range_push()

# torch.profiler.tensorboard_trace_handler
result = torch.profiler.tensorboard_trace_handler()

# torch.promote_types
result = torch.promote_types()

# torch.quantize_per_channel
result = torch.quantize_per_channel()

# torch.quantize_per_tensor
result = torch.quantize_per_tensor()

# torch.quantized_batch_norm
result = torch.quantized_batch_norm()

# torch.quantized_max_pool1d
result = torch.quantized_max_pool1d()

# torch.quantized_max_pool2d
result = torch.quantized_max_pool2d()

# torch.quasirandom.SobolEngine
result = torch.quasirandom.SobolEngine

# torch.random.fork_rng
result = torch.random.fork_rng()

# torch.resolve_conj
result = torch.resolve_conj()

# torch.resolve_neg
result = torch.resolve_neg()

# torch.result_type
result = torch.result_type()

# torch.set_deterministic_debug_mode
result = torch.set_deterministic_debug_mode(True)

# torch.set_float32_matmul_precision
result = torch.set_float32_matmul_precision(True)

# torch.set_flush_denormal
result = torch.set_flush_denormal(True)

# torch.set_warn_always
result = torch.set_warn_always(True)

# torch.signal.windows.bartlett
result = torch.signal.windows.bartlett()

# torch.signal.windows.kaiser
result = torch.signal.windows.kaiser()

# torch.signal.windows.nuttall
result = torch.signal.windows.nuttall()

# torch.smm
result = torch.smm()

# torch.sparse.as_sparse_gradcheck
result = torch.sparse.as_sparse_gradcheck()

# torch.sparse.check_sparse_tensor_invariants
result = torch.sparse.check_sparse_tensor_invariants()

# torch.sparse.log_softmax
result = torch.sparse.log_softmax()

# torch.sparse.sampled_addmm
result = torch.sparse.sampled_addmm()

# torch.sparse.spdiags
result = torch.sparse.spdiags()

# torch.sparse_bsc_tensor
result = torch.sparse_bsc_tensor()

# torch.sparse_bsr_tensor
result = torch.sparse_bsr_tensor()

# torch.sparse_compressed_tensor
result = torch.sparse_compressed_tensor()

# torch.sparse_csc_tensor
result = torch.sparse_csc_tensor()

# torch.special.airy_ai
result = torch.special.airy_ai()

# torch.special.bessel_j0
result = torch.special.bessel_j0()

# torch.special.bessel_j1
result = torch.special.bessel_j1()

# torch.special.entr
result = torch.special.entr()

# torch.special.log_ndtr
result = torch.special.log_ndtr()

# torch.special.scaled_modified_bessel_k0
result = torch.special.scaled_modified_bessel_k0()

# torch.special.scaled_modified_bessel_k1
result = torch.special.scaled_modified_bessel_k1()

# torch.special.spherical_bessel_j0
result = torch.special.spherical_bessel_j0()

# torch.special.zeta
result = torch.special.zeta()

# torch.sspaddmm
result = torch.sspaddmm()

# torch.sym_float
result = torch.sym_float()

# torch.sym_int
result = torch.sym_int()

# torch.sym_ite
result = torch.sym_ite()

# torch.sym_max
result = torch.sym_max()

# torch.sym_min
result = torch.sym_min()

# torch.sym_not
result = torch.sym_not()

# torch.unravel_index
result = torch.unravel_index()

# torch.use_deterministic_algorithms
result = torch.use_deterministic_algorithms()

# torch.utils.benchmark.CallgrindStats
result = torch.utils.benchmark.CallgrindStats

# torch.utils.benchmark.FunctionCounts
result = torch.utils.benchmark.FunctionCounts

# torch.utils.benchmark.Measurement
result = torch.utils.benchmark.Measurement

# torch.utils.benchmark.Timer
result = torch.utils.benchmark.Timer

# torch.utils.checkpoint.checkpoint_sequential
result = torch.utils.checkpoint.checkpoint_sequential()

# torch.utils.checkpoint.set_checkpoint_debug_enabled
result = torch.utils.checkpoint.set_checkpoint_debug_enabled(True)

# torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version
result = torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version()

# torch.utils.cpp_extension.include_paths
result = torch.utils.cpp_extension.include_paths()

# torch.utils.cpp_extension.is_ninja_available
result = torch.utils.cpp_extension.is_ninja_available()

# torch.utils.cpp_extension.load_inline
result = torch.utils.cpp_extension.load_inline()

# torch.utils.cpp_extension.verify_ninja_availability
result = torch.utils.cpp_extension.verify_ninja_availability()

# torch.utils.data.StackDataset
result = torch.utils.data.StackDataset

# torch.utils.data._utils.collate.collate
result = torch.utils.data._utils.collate.collate()

# torch.utils.data.default_convert
result = torch.utils.data.default_convert()

# torch.utils.generate_methods_for_privateuse1_backend
result = torch.utils.generate_methods_for_privateuse1_backend()

# torch.utils.get_cpp_backtrace
result = torch.utils.get_cpp_backtrace()

# torch.utils.mobile_optimizer.optimize_for_mobile
result = torch.utils.mobile_optimizer.optimize_for_mobile()

# torch.utils.rename_privateuse1_backend
result = torch.utils.rename_privateuse1_backend()

# torch.utils.swap_tensors
result = torch.utils.swap_tensors()

# torch.utils.tensorboard.writer.SummaryWriter
result = torch.utils.tensorboard.writer.SummaryWriter

# torch.vmap
result = torch.vmap()

# torch.xpu.Event
result = torch.xpu.Event

# torch.xpu.StreamContext
result = torch.xpu.StreamContext

# torch.xpu.Stream
result = torch.xpu.Stream

# torch.xpu.current_device
result = torch.xpu.current_device()

# torch.xpu.current_stream
result = torch.xpu.current_stream()

# torch.xpu.device
result = torch.xpu.device()

# torch.xpu.device_count
result = torch.xpu.device_count()

# torch.xpu.device_of
x = torch.randn(2, 3, 4)
result = torch.xpu.device_of(x)

# torch.xpu.empty_cache
result = torch.xpu.empty_cache()

# torch.xpu.get_device_capability
result = torch.xpu.get_device_capability()

# torch.xpu.get_device_name
result = torch.xpu.get_device_name()

# torch.xpu.get_device_properties
result = torch.xpu.get_device_properties()

# torch.xpu.get_rng_state
result = torch.xpu.get_rng_state()

# torch.xpu.get_rng_state_all
result = torch.xpu.get_rng_state_all()

# torch.xpu.init
result = torch.xpu.init()

# torch.xpu.initial_seed
result = torch.xpu.initial_seed()

# torch.xpu.is_available
result = torch.xpu.is_available()

# torch.xpu.is_initialized
result = torch.xpu.is_initialized()

# torch.xpu.manual_seed
result = torch.xpu.manual_seed(2024)

# torch.xpu.manual_seed_all
result = torch.xpu.manual_seed_all()

# torch.xpu.seed
result = torch.xpu.seed()

# torch.xpu.seed_all
result = torch.xpu.seed_all()

# torch.xpu.set_device
result = torch.xpu.set_device(True)

# torch.xpu.set_rng_state
result = torch.xpu.set_rng_state(True)

# torch.xpu.set_rng_state_all
result = torch.xpu.set_rng_state_all(True)

# torch.xpu.set_stream
result = torch.xpu.set_stream(True)

# torch.xpu.stream
result = torch.xpu.stream()

# torch.xpu.synchronize
result = torch.xpu.synchronize()
