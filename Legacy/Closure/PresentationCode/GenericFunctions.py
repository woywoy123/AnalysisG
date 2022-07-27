from Functions.IO.Files import WriteDirectory, Directories
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.Event.CacheGenerators import Generate_Cache_Batches

def CreateWorkspace(Name, FileDir, Cache, Stop = 100):
    if Cache:
        x = WriteDirectory()
        x.MakeDir("PresentationPlots/" + Name)
    return Generate_Cache_Batches(FileDir, Stop = Stop, OutDirectory = "PresentationPlots/" + Name, CreateCache = Cache)
    
def BackupData(Dir, DB = None, Name = None, restore = False):
    if restore == False:
        PickleObject(DB, Name, Dir)
    else:
        Files = Directories(Dir).ListFilesInDir(Dir)
        Backup = {}
        for i in Files:
            b = UnpickleObject(i, Dir)
            for k, j in b.items():
                if k not in Backup:
                    Backup[k] = []
                Backup[k] += j
        return Backup

def Mass(child):
    from Functions.Particles.Particles import Particle
    m = Particle()
    m.CalculateMass(child)
    return m.Mass_GeV
