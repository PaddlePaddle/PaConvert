import paddle

paddle.enable_compat(level=2)


class PT_Optimizer(paddle.optim.Optimizer):
    pass


PT_Optimizer.load_state_dict(self, swa_state_dict)
PT_Optimizer.load_state_dict(self, swa_state_dict, True)
self.optimizer.load_state_dict(opt_state_dict)
sgd.load_state_dict(opt_state_dict)
