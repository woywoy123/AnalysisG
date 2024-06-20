# distutils: language=c++
# cython: language_level=3

from libcpp cimport float
from AnalysisG.core.tools cimport *
from AnalysisG.core.lossfx cimport *

cdef class OptimizerConfig:

    def __cinit__(self): self.params = new optimizer_params_t()
    def __init__(self): pass
    def __dealloc__(self): del self.params

    @property
    def Optimizer(self): return env(self.params.optimizer)

    @Optimizer.setter
    def Optimizer(self, str val): self.params.optimizer = enc(val)

    @property
    def lr(self): return self.params.lr

    @lr.setter
    def lr(self, double val): self.params.lr = val

    @property
    def lr_decay(self): return self.params.lr_decay

    @lr_decay.setter
    def lr_decay(self, double val): self.params.lr_decay = val

    @property
    def weight_decay(self): return self.params.weight_decay

    @weight_decay.setter
    def weight_decay(self, double val): self.params.weight_decay = val

    @property
    def initial_accumulator_value(self): return self.params.initial_accumulator_value

    @initial_accumulator_value.setter
    def initial_accumulator_value(self, double val): self.params.initial_accumulator_value = val


    @property
    def eps(self): return self.params.eps

    @eps.setter
    def eps(self, double val): self.params.eps = val


    @property
    def tolerance_grad(self): return self.params.tolerance_grad

    @tolerance_grad.setter
    def tolerance_grad(self, double val): self.params.tolerance_grad = val


    @property
    def tolerance_change(self): return self.params.tolerance_change

    @tolerance_change.setter
    def tolerance_change(self, double val): self.params.tolerance_change = val


    @property
    def alpha(self): return self.params.alpha

    @alpha.setter
    def alpha(self, double val): self.params.alpha = val

    @property
    def momentum(self): return self.params.momentum

    @momentum.setter
    def momentum(self, double val): self.params.momentum = val

    @property
    def dampening(self): return self.params.dampening

    @dampening.setter
    def dampening(self, double val): self.params.dampening = val


    @property
    def amsgrad(self): return self.params.amsgrad

    @amsgrad.setter
    def amsgrad(self, bool val): self.params.amsgrad = val

    @property
    def centered(self): return self.params.centered

    @centered.setter
    def centered(self, bool val): self.params.centered = val

    @property
    def nesterov(self): return self.params.nesterov

    @nesterov.setter
    def nesterov(self, bool val): self.params.nesterov = val

    @property
    def max_iter(self): return self.params.max_iter

    @max_iter.setter
    def max_iter(self, int val): self.params.max_iter = val

    @property
    def max_eval(self): return self.params.max_eval

    @max_eval.setter
    def max_eval(self, int val): self.params.max_eval = val

    @property
    def history_size(self): return self.params.history_size

    @history_size.setter
    def history_size(self, int val): self.params.history_size = val

    @property
    def betas(self): return self.params.beta_hack

    @betas.setter
    def betas(self, tuple val):
        self.params.beta_hack = vector[float](val[0], val[1])

