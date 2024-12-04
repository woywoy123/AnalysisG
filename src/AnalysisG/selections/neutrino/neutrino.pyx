# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.core.tools cimport *

cdef class Neutrino(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new neutrino()
        self.tt = <neutrino*>self.ptr

    def __dealloc__(self): del self.tt

    cdef void transform_dict_keys(self):
        self.delta_met   = as_basic_dict(&self.tt.delta_met  )
        self.delta_metnu = as_basic_dict(&self.tt.delta_metnu)
        self.obs_met     = as_basic_dict(&self.tt.obs_met    )
        self.nus_met     = as_basic_dict(&self.tt.nus_met    )
        self.dist_nu     = as_basic_dict(&self.tt.dist_nu    )

        self.pdgid       = as_dict(&self.tt.pdgid      )
        self.tru_topmass = as_dict(&self.tt.tru_topmass)
        self.tru_wmass   = as_dict(&self.tt.tru_wmass  )

        self.exp_topmass = as_dict(&self.tt.exp_topmass)
        self.exp_wmass   = as_dict(&self.tt.exp_wmass  )
