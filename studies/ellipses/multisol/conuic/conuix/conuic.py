from conuix.attestation.base import attestation
from conuix.types.base import structs_t
from conuix.visuals.plots import *

import numpy as np
np.set_printoptions(linewidth=20000)

class conuic(structs_t, attestation, figures):

    def __init__(self, bq, lp, nu, pl, nm): 
        attestation.__init__(self)
        structs_t.__init__(self, bq, lp)

        # ----- truth ----- #
        # Closure testing #
        self.nu = nu
        self.pls = pl
        self.mns = nm

        self.proof_base_relation()
        self.proof_pcl1_relation()
        self.proof_pcl1_eigen()
        self.proof_pcl2_relation()
        #self.proof_affine_relation()
        self.surfaces()
        self.debug()
        #------------------#
     
    def debug(self):
        #self.Transformation()

        pass


