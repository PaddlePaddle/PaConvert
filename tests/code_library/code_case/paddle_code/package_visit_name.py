import paddle
import paddleformers

paddle.enable_compat(level=2)
setattr(paddle, "nn", nn_mymodule)
hasattr(paddle, "nn")
hasattr(paddle, "__version__")
hasattr(paddleformers, "__version__")
