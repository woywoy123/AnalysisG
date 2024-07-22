from .mc16 import *

class samples:
    def __init__(self, path, mc):
        self._path = path
        self._mc = mc

    def sample(self, name):
        data = []
        if name == "sm_tttt":        data += sm_tttt()._data
        if name == "bsm_ttttH_400":  data += bsm_tttH_400()._data
        if name == "bsm_ttttH_500":  data += bsm_tttH_500()._data
        if name == "bsm_ttttH_600":  data += bsm_tttH_600()._data
        if name == "bsm_ttttH_700":  data += bsm_tttH_700()._data
        if name == "bsm_ttttH_800":  data += bsm_tttH_800()._data
        if name == "bsm_ttttH_900":  data += bsm_tttH_900()._data
        if name == "bsm_ttttH_1000": data += bsm_tttH_1000()._data
        if name == "sm_ttbar":       data += sm_ttbar()._data
        if name == "sm_tt_Vll":      data += sm_tt_Vll()._data
        if name == "sm_ttH":         data += sm_ttH()._data
        if name == "sm_t":           data += sm_t()._data
        if name == "sm_other":       data += sm_other()._data
        if not len(data): data += [name]
        return [self._path + "/" + data[i] for i in range(len(data))]
