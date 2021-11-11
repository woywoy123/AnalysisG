from Functions.Event.Event import EventGenerator
from Functions.IO.IO import PickleObject, UnpickleObject

events = -1
def TestSingleFile():
    dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
    
    ev = EventGenerator(dir, DebugThresh = events)
    ev.SpawnEvents()
    ev.CompileEvent()
    PickleObject(ev, "SingleFile")
    #ev = UnpickleObject("SingleFile")
