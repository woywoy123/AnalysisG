from .Hashing import Hashing 

class ROOTFile(Hashing):
    
    def __init__(self, Filename):
        self.HashMap = {}
        self.Filename = Filename
        self.EventMap = {}
        self._lock = False
    
    def MakeHash(self):
        for i in self.EventMap:
            _hash = self.MD5(self.Filename + "/" + str(self.EventMap[i].EventIndex))
            self.EventMap[i].Filename = _hash
            self.HashMap[_hash] = self.EventMap[i]

    def AddEvent(self, Event):
        self.EventMap[Event.EventIndex] = Event
    
    def __len__(self):
        return len(self.EventMap)

    def list(self):
        return list(self.EventMap.values())
    
    def hash(self):
        if len(self) != len(self.HashMap) and self._lock == False:
            self.MakeHash()
        return list(self.HashMap)
    
    def __getitem__(self, key):
        if key in self.HashMap:
            return self.HashMap[key]
        if key in self.EventMap:
            return self.EventMap[key]
        return False
    
    def __setitem__(self, key, obj):
        if key in self.hash() and self._lock == False:
            self.EventMap[self.HashMap[key].EventIndex] 
            return self

        if key in self.EventMap and self._lock == False:
            self.HashMap[self.EventMap[key].Filename] = obj
            self.EventMap[key] = obj
            return self
         
    def __contains__(self, key):
        self.hash()
        if key in self.HashMap:
            return True 
        if key in self.EventMap:
            return True
        return False

    def __add__(self, other):
        hashes = self.hash() + other.hash() 
        evnts = [ self.HashMap[_hash] for _hash in self.HashMap ] + [ other.HashMap[_hash] for _hash in other.HashMap ]
        
        self.HashMap = {}
        for _hash, evnt in zip(hashes, evnts):
            if _hash not in self.HashMap:
                self.HashMap[_hash] = evnt
                continue
            self.HashMap[_hash] += evnt

        self.EventMap = self.EventMap if self._lock else {i : ev.UpdateIndex(i) for i, ev in zip(range(len(self.HashMap)), self.HashMap.values())}
        return self
    
    def ClearEvents(self):
        if self._lock:
            return self
        self._lock = True
        self.HashMap = {i.Filename : "" for i in self.HashMap.values()}
        self.EventMap = {i.EventIndex : "" for i in self.EventMap.values()}

    def RestoreEvents(self, events):
        if isinstance(events, dict):
            events = list(events.values())    
        self.HashMap |= { i.Filename : i for i in events if i.Filename in self.HashMap }
        self.EventMap |= { i.EventIndex : i for i in events if i.EventIndex in self.EventMap}
        self._lock = False
