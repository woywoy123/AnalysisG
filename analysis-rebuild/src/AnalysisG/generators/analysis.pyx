# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.tools cimport *
from AnalysisG.core.event_template cimport *
from AnalysisG.core.graph_template cimport *
from AnalysisG.core.model_template cimport *
from AnalysisG.core.lossfx cimport *
from AnalysisG.generators.analysis cimport analysis


cdef class Analysis:

    def __cinit__(self):
        self.ana = new analysis()

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.ana

    def AddSamples(self, str path, str label):
        self.ana.add_samples(enc(path), enc(label))

    def AddEvent(self, EventTemplate ev, str label):
        self.ana.add_event_template(ev.ptr, enc(label))

    def AddGraph(self, GraphTemplate ev, str label):
        self.ana.add_graph_template(ev.ptr, enc(label))

    def AddModel(self, ModelTemplate model, OptimizerConfig op, str run_name):
        self.ana.add_model(model.nn_ptr, op.params, enc(run_name))

    def Start(self):
        self.ana.start()

    @property
    def OutputPath(self): return env(self.ana.output_path)

    @OutputPath.setter
    def OutputPath(self, str path): self.ana.output_path = enc(path)

    @property
    def kFolds(self): return self.ana.kfolds

    @kFolds.setter
    def kFolds(self, int k): self.ana.kfolds = k

    @property
    def Epochs(self): return self.ana.epochs

    @Epochs.setter
    def Epochs(self, int k): self.ana.epochs = k

    @property
    def NumExamples(self): return self.ana.num_examples

    @NumExamples.setter
    def NumExamples(self, int k): self.ana.num_examples = k

    @property
    def TrainSize(self): return self.ana.train_size

    @TrainSize.setter
    def TrainSize(self, float k): self.ana.train_size = k


