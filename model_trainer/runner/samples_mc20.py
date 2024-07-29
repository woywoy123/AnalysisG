from .mc20 import *

class samples:
    def __init__(self, path, mc):
        self._path = path
        self._mc = mc

    def sample(self, name):
        data = []
        if name == "sm_tttt":         data+=sm_tttt()._data
        if name == "sm_ttt":          data+=sm_ttt()._data
        if name == "bsm_ttttH_400":   data+=bsm_ttttH_400()._data
        if name == "bsm_ttttH_500":   data+=bsm_ttttH_500()._data
        if name == "bsm_ttttH_600":   data+=bsm_ttttH_600()._data
        if name == "bsm_ttttH_700":   data+=bsm_ttttH_700()._data
        if name == "bsm_ttttH_800":   data+=bsm_ttttH_800()._data
        if name == "bsm_ttttH_900":   data+=bsm_ttttH_900()._data
        if name == "bsm_ttttH_1000":  data+=bsm_ttttH_1000()._data
        if name == "sm_ttbar":        data+=sm_ttbar()._data
        if name == "sm_ttV":          data+=sm_ttV()._data
        if name == "sm_tt_Vll":       data+=sm_tt_Vll()._data
        if name == "sm_Vll":          data+=sm_Vll()._data
        if name == "sm_llgammagamma": data+=sm_llgammagamma()._data
        if name == "sm_ttH":          data+=sm_ttH()._data
        if name == "sm_t":            data+=sm_t()._data
        if name == "sm_wh":           data+=sm_wh()._data
        if name == "sm_VVll":         data+=sm_VVll()._data
        if name == "sm_llll":         data+=sm_llll()._data
        if not len(data): data += [name]
        return [self._path + "/" + data[i] for i in range(len(data))]
