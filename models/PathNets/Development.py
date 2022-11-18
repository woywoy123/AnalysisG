from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event

if __name__ == "__main__":
   
    direc = "/CERN/Samples/Processed/bsm4tops"


    Ana = Analysis()
    Ana.Event = Event 
    Ana.EventCache = True 
    Ana.InputSample("bsm4top", direc + "/mc16a/DAOD_TOPQ1.21955713._000001.root")
    Ana.Threads = 1
    Ana.Launch()
