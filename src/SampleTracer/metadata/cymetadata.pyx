# distuils: language = c++
# cython: language_level = 3

from cymetadata cimport CyMetaData
from libcpp.string cimport string
from libcpp cimport bool

try:
    import pyAMI.client
    import pyAMI.atlas.api as atlas
except ModuleNotFoundError: pass

from uproot.exceptions import KeyInFileError
import warnings
import uproot
import json
import signal

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class MetaData:

    cdef CyMetaData* ptr
    cdef bool loaded
    cdef public client
    cdef public sampletype

    def __cinit__(self):
        self.ptr = new CyMetaData()
        self.client = None

    def __init__(self):
        warnings.filterwarnings("ignore")
        self.loaded = True
        try: self.client = pyAMI.client.Client("atlas")
        except: self.loaded = False
        self.sampletype = "DAOD_TOPQ1"

    def get(self, dict command) -> dict:
        cdef dict x, out
        cdef str l
        if not self.loaded: return {}
        x = [k for k in uproot.iterate(**command)][0]

        out = {}
        for l in command["expressions"]:
            try: out[l] = x[l].tolist()[0]
            except KeyError: print(l)
        return out

    def file_data(self, str i) -> None:
        cdef dict command = {
                "files" : i + ":sumWeights",
                "expressions" :  ["dsid", "AMITag", "generators"],
                "how" : dict, "library" : "np"
        }

        cdef out = self.get(command)
        try: self.ptr.dsid = out["dsid"]
        except KeyError: pass

        try: self.ptr.AMITag = enc(out["AMITag"])
        except KeyError: pass

        try: self.ptr.generators = enc(out["generators"].replace("+", " "))
        except KeyError: pass

    def file_tracker(self, str i) -> void:
        cdef dict command = {
                "files" : i + ":AnalysisTracking",
                "expressions" : ["jsonData"],
                "step_size" : 100, "how" : dict,
                "library" : "np"
        }

        cdef list f = self.get(command)["jsonData"].split("\n")
        if len(f) == 0: return

        cdef str out = "\n".join([k for k in f if "BtagCDIPath" not in k])
        cdef int index = 0
        cdef dict this

        t = json.loads(out)
        this = t["inputConfig"]
        self.ptr.isMC = this["isMC"]
        self.ptr.derivationFormat = enc(this["derivationFormat"])

        this = t["configSettings"]
        for x in this: self.ptr.addconfig(enc(x), enc(this[x]))

        for x in t["inputFiles"]:
            self.ptr.addsamples(index, enc(x[0]))
            index += x[1]

    def file_truth(self, str i):
        cdef dict command = {
                "files" : i + ":truth",
                "expressions" : ["eventNumber"],
                "how" : dict, "library" : "np"
        }
        try: self.ptr.eventNumber = self.get(command)["eventNumber"]
        except KeyError: pass

    def search(self):
        def _sig(signum, frame): return ""
        if self.client is None: return
        signal.signal(signal.SIGALRM, _sig)
        signal.alarm(10)
        warnings.filterwarnings("ignore")
        print(atlas.list_datasets(
                self.client,
                dataset_number = [str(self.ptr.dsid)],
                type = self.sampletype
        ))






