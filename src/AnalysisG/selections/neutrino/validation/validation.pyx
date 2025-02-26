# distutils: language=c++
# cython: language_level=3

from AnalysisG.core.selection_template cimport *
from AnalysisG.core.particle_template cimport *
from AnalysisG.core.tools cimport *

cdef class Neutrino(ParticleTemplate):
    def __cinit__(self): self.ptr = new nu()
    def __dealloc__(self): del self.ptr

    @property
    def distance(self):
        cdef nu* lx = <nu*>(self.ptr)
        return lx.distance

    @distance.setter
    def distance(self, float v):
        cdef nu* lx = <nu*>(self.ptr)
        lx.distance = v

cdef class TopQ(ParticleTemplate):
    def __cinit__(self): self.ptr = new tquark()
    def __dealloc__(self): del self.ptr

cdef class BottomQ(ParticleTemplate):
    def __cinit__(self): self.ptr = new bquark()
    def __dealloc__(self): del self.ptr

cdef class Lepton(ParticleTemplate):
    def __cinit__(self): self.ptr = new lepton()
    def __dealloc__(self): del self.ptr

cdef class Boson(ParticleTemplate):
    def __cinit__(self): self.ptr = new boson()
    def __dealloc__(self): del self.ptr

cdef class Validation(SelectionTemplate):
    def __cinit__(self):
        self.ptr = new validation()
        self.tt = <validation*>self.ptr

        self.met = []
        self.phi = []

        self.truth_nus = []
        self.truth_tops = []
        self.truth_bosons = []

        self.reco_leptons = []
        self.reco_bosons = []

        self.truth_leptons = []
        self.truth_bquarks = []

        self.truth_bjets = []
        self.truth_jets_top = []

        self.bjets = []
        self.jets_top = []
        self.lepton_jets_top = []

        self.c1_reconstructed_children_nu = []
        self.c1_reconstructed_truthjet_nu = []
        self.c1_reconstructed_jetchild_nu = []
        self.c1_reconstructed_jetlep_nu = []

        self.c2_reconstructed_children_nu = []
        self.c2_reconstructed_truthjet_nu = []
        self.c2_reconstructed_jetchild_nu = []
        self.c2_reconstructed_jetlep_nu = []

    def __dealloc__(self): del self.tt

    cdef list build_p0(self, vector[tquark*]* inpt):
        cdef int i
        cdef TopQ txp
        cdef list out = []
        for i in range(inpt.size()):
            txp = TopQ()
            txp.set_particle(inpt.at(i))
            out.append(txp)
        return out

    cdef list build_p1(self, vector[bquark*]* inpt):
        cdef int i
        cdef BottomQ txp
        cdef list out = []
        for i in range(inpt.size()):
            txp = BottomQ()
            txp.set_particle(inpt.at(i))
            out.append(txp)
        return out

    cdef list build_p2(self, vector[lepton*]* inpt):
        cdef int i
        cdef Lepton txp
        cdef list out = []
        for i in range(inpt.size()):
            txp = Lepton()
            txp.set_particle(inpt.at(i))
            out.append(txp)
        return out

    cdef list  build_p3(self, vector[boson*]* inpt):
        cdef int i
        cdef Boson txp
        cdef list out = []
        for i in range(inpt.size()):
            txp = Boson()
            txp.set_particle(inpt.at(i))
            out.append(txp)
        return out

    cdef list build_p4(self, vector[nu*]* inpt):
        cdef int i
        cdef Neutrino txp
        cdef list out = []
        for i in range(inpt.size()):
            txp = Neutrino()
            txp.set_particle(inpt.at(i))
            out.append(txp)
        return out

    cdef void transform_dict_keys(self):
        cdef pair[string, package] itx
        cdef package* pkg

        for itx in self.tt.data_out:
            pkg = &itx.second
            self.met.append(pkg.met)
            self.phi.append(pkg.phi)

            self.truth_nus.append(   self.build_p4(&pkg.truth_nus))
            self.truth_tops.append(  self.build_p0(&pkg.truth_tops))
            self.truth_bosons.append(self.build_p3(&pkg.truth_bosons))

            self.reco_leptons.append(self.build_p2(&pkg.reco_leptons))
            self.reco_bosons.append(self.build_p3(&pkg.reco_bosons))

            self.truth_leptons.append(self.build_p2(&pkg.truth_leptons))
            self.truth_bquarks.append(self.build_p1(&pkg.truth_bquarks))

            self.truth_bjets.append(self.build_p1(&pkg.truth_bjets))
            self.truth_jets_top.append(self.build_p0(&pkg.truth_jets_top))

            self.bjets.append(self.build_p1(&pkg.bjets))
            self.jets_top.append(self.build_p0(&pkg.jets_top))
            self.lepton_jets_top.append(self.build_p0(&pkg.lepton_jets_top))

            self.c1_reconstructed_children_nu.append(self.build_p4(&pkg.c1_reconstructed_children_nu))
            self.c1_reconstructed_truthjet_nu.append(self.build_p4(&pkg.c1_reconstructed_truthjet_nu))
            self.c1_reconstructed_jetchild_nu.append(self.build_p4(&pkg.c1_reconstructed_jetchild_nu))
            self.c1_reconstructed_jetlep_nu.append(self.build_p4(&pkg.c1_reconstructed_jetlep_nu))

            self.c2_reconstructed_children_nu.append(self.build_p4(&pkg.c2_reconstructed_children_nu))
            self.c2_reconstructed_truthjet_nu.append(self.build_p4(&pkg.c2_reconstructed_truthjet_nu))
            self.c2_reconstructed_jetchild_nu.append(self.build_p4(&pkg.c2_reconstructed_jetchild_nu))
            self.c2_reconstructed_jetlep_nu.append(self.build_p4(&pkg.c2_reconstructed_jetlep_nu))
