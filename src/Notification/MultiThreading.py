from .Notification import Notification

class MultiThreading(Notification):

    def __init__(self):
       pass

    def StartingJobs(self, ith):
        self.Success("!!STARTED JOB " + str(ith+1) + "/" + str(len(self._chnk)))
    
    def FinishedJobs(self, w):
        self.Success("!!WORKER FINISHED " + str(w) + "/" + str(self._threads))

    def RecoveredThread(self, w):
        self.Warning("A Threads has failed. Switching to main thread. Worker: " + str(w))

    def AlertOnEmptyList(self):
        self._lock = False
        if len(self._lists) == 0:
            self.Failure("Can't process an empty list. Skipping...")
            self._lock = True
