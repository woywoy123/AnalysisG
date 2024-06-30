# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport pair, map

from AnalysisG.core.tools cimport *
from AnalysisG.core.event_template cimport *
from AnalysisG.core.graph_template cimport *
from AnalysisG.core.model_template cimport *
from AnalysisG.core.lossfx cimport *
from AnalysisG.generators.analysis cimport analysis

from time import sleep
from tqdm import tqdm

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
        def factory(title):
            pdar = tqdm(
                    total = 100, desc = title,
                    leave = False,
                    dynamic_ncols = True
            )
            return pdar


        self.ana.start()

        cdef dict bars = {}
        cdef pair[string, float] itr
        cdef pair[string, bool] itb

        cdef map[string, float] o = self.ana.progress()
        cdef map[string, string] desc = self.ana.progress_mode()
        cdef map[string, string] repo = self.ana.progress_report()
        cdef map[string, string] updr = repo

        cdef map[string, bool] compl
        cdef map[string, bool] same

        for itr in o:
            bars[itr.first] = factory(env(itr.first))
            same[itr.first] = True

        while True:
            desc = self.ana.progress_mode()
            updr = self.ana.progress_report()
            o    = self.ana.progress()
            compl = self.ana.is_complete()
            for itr in o:
                bars[itr.first].update(itr.second - bars[itr.first].n)
                bars[itr.first].set_description(env(desc[itr.first]))
                bars[itr.first].refresh()
                same[itr.first] = updr[itr.first] == repo[itr.first]
            sleep(0.1)

            c = 0
            for itb in compl: c += compl[itb.first]
            if c == compl.size(): break

            x = 0
            for itb in same: x += not itb.second
            if x != same.size(): continue

            repo = updr
            for itr in o: bars[itr.first].close(); del bars[itr.first]

            print("========= Model Report =========")
            for itr in o: print(env(updr[itr.first]))
            print("========= END Report =========")
            for itr in o:
                bars[itr.first] = factory(env(desc[itr.first]))
                same[itr.first] = updr[itr.first] == repo[itr.first]
        self.ana.attach_threads()

    @property
    def OutputPath(self): return env(self.ana.m_settings.output_path)

    @OutputPath.setter
    def OutputPath(self, str path): self.ana.m_settings.output_path = enc(path)

    @property
    def kFolds(self): return self.ana.m_settings.kfolds

    @kFolds.setter
    def kFolds(self, int k): self.ana.m_settings.kfolds = k

    @property
    def Epochs(self): return self.ana.m_settings.epochs

    @Epochs.setter
    def Epochs(self, int k): self.ana.m_settings.epochs = k

    @property
    def NumExamples(self): return self.ana.m_settings.num_examples

    @NumExamples.setter
    def NumExamples(self, int k): self.ana.m_settings.num_examples = k

    @property
    def TrainSize(self): return self.ana.m_settings.train_size

    @TrainSize.setter
    def TrainSize(self, float k): self.ana.m_settings.train_size = k

    @property
    def Training(self): return self.ana.m_settings.training

    @Training.setter
    def Training(self, bool val): self.ana.m_settings.training = val

    @property
    def Validation(self): return self.ana.m_settings.validation

    @Validation.setter
    def Validation(self, bool val): self.ana.m_settings.validation = val

    @property
    def Evaluation(self): return self.ana.m_settings.evaluation

    @Evaluation.setter
    def Evaluation(self, bool val): self.ana.m_settings.evaluation = val

    @property
    def ContinueTraining(self): return self.ana.m_settings.continue_training

    @ContinueTraining.setter
    def ContinueTraining(self, bool val): self.ana.m_settings.continue_training = val

    @property
    def nBins(self): return self.ana.m_settings.nbins

    @nBins.setter
    def nBins(self, int val): self.ana.m_settings.nbins = val

    @property
    def Refresh(self): return self.ana.m_settings.refresh

    @Refresh.setter
    def Refresh(self, int val): self.ana.m_settings.refresh = val

    @property
    def MaxRange(self): return self.ana.m_settings.max_range

    @MaxRange.setter
    def MaxRange(self, int val): self.ana.m_settings.max_range = val

    @property
    def VarPt(self): return self.ana.m_settings.var_pt

    @VarPt.setter
    def VarPt(self, str val): self.ana.m_settings.var_pt = enc(val)

    @property
    def VarEta(self): return self.ana.m_settings.var_eta

    @VarEta.setter
    def VarEta(self, str val): self.ana.m_settings.var_eta = enc(val)

    @property
    def VarPhi(self): return self.ana.m_settings.var_phi

    @VarPhi.setter
    def VarPhi(self, str val): self.ana.m_settings.var_phi = enc(val)

    @property
    def VarEnergy(self): return self.ana.m_settings.var_energy

    @VarEnergy.setter
    def VarEnergy(self, str val): self.ana.m_settings.var_energy = enc(val)

    @property
    def Targets(self): return env_vec(&self.ana.m_settings.targets)

    @Targets.setter
    def Targets(self, list val): self.ana.m_settings.targets = enc_list(val)

    @property
    def DebugMode(self): return self.ana.m_settings.debug_mode

    @DebugMode.setter
    def DebugMode(self, bool val): self.ana.m_settings.debug_mode = val