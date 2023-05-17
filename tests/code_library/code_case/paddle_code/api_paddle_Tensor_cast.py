import paddle

print("#########################case1#########################")
cpu = "cpu"
a = paddle.randn(shape=[2, 3])
c = paddle.randn(shape=[2, 3], dtype="float64")
if isinstance(cpu, paddle.dtype):
    dtype = cpu
elif isinstance(cpu, str) and cpu not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = cpu
elif isinstance(cpu, paddle.Tensor):
    dtype = cpu.dtype
else:
    dtype = a.dtype
b = a.cast(dtype)
print("#########################case2#########################")
if isinstance("cpu", paddle.dtype):
    dtype = "cpu"
elif isinstance("cpu", str) and "cpu" not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = "cpu"
elif isinstance("cpu", paddle.Tensor):
    dtype = "cpu".dtype
else:
    dtype = a.dtype
b = a.cast(dtype)
print("#########################case3#########################")
b = a.cast("float64")
print("#########################case4#########################")
if isinstance("float64", paddle.dtype):
    dtype = "float64"
elif isinstance("float64", str) and "float64" not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = "float64"
elif isinstance("float64", paddle.Tensor):
    dtype = "float64".dtype
else:
    dtype = a.dtype
b = a.cast(dtype)
print("#########################case5#########################")
b = a.cast("float64")
print("#########################case6#########################")
if isinstance(c, paddle.dtype):
    dtype = c
elif isinstance(c, str) and c not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = c
elif isinstance(c, paddle.Tensor):
    dtype = c.dtype
else:
    dtype = a.dtype
b = a.cast(dtype)
print("#########################case7#########################")
if isinstance("float16", paddle.dtype):
    dtype = "float16"
elif isinstance("float16", str) and "float16" not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = "float16"
elif isinstance("float16", paddle.Tensor):
    dtype = "float16".dtype
else:
    dtype = a.dtype
a = a.cast(dtype)
print("#########################case8#########################")
table = a
if isinstance(table.place, paddle.dtype):
    dtype = table.place
elif isinstance(table.place, str) and table.place not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = table.place
elif isinstance(table.place, paddle.Tensor):
    dtype = table.place.dtype
else:
    dtype = a.dtype
b = a.cast(dtype)
print("#########################case9#########################")
if isinstance("float32", paddle.dtype):
    dtype = "float32"
elif isinstance("float32", str) and "float32" not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = "float32"
elif isinstance("float32", paddle.Tensor):
    dtype = "float32".dtype
else:
    dtype = a.dtype
b = a.cast(dtype)
print("#########################case10#########################")
device = "cpu"
if isinstance("bool", paddle.dtype):
    dtype = "bool"
elif isinstance("bool", str) and "bool" not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = "bool"
elif isinstance("bool", paddle.Tensor):
    dtype = "bool".dtype
else:
    dtype = paddle.to_tensor(data=[-1]).dtype
b = paddle.to_tensor(data=[-1]).cast(dtype)
print("#########################case11#########################")
dtype = "float32"
b = a.cast(dtype)
print("#########################case12#########################")
if isinstance("cpu", paddle.dtype):
    dtype = "cpu"
elif isinstance("cpu", str) and "cpu" not in ["cpu", "cuda", "ipu", "xpu"]:
    dtype = "cpu"
elif isinstance("cpu", paddle.Tensor):
    dtype = "cpu".dtype
else:
    dtype = a.dtype
b = a.cast(dtype)
