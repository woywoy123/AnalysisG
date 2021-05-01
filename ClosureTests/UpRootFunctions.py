from BaseFunctions.UpRootFunctions import FileObjectsToArrays
from BaseFunctions.IO import ObjectsFromFile


def TestFileObjectsToArrays():
   
    files = "/CERN/Grid/Samples/user.tnommens.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r10201_p4174.bsm4t-21.2.102-4-0-mc16d_output_root/user.tnommens.24703615._000001.output.root"
    Entry_Objects = ObjectsFromFile(files, "nominal", ["top_FromRes", "truth_top_pt"])
    FileObjectsToArrays(Entry_Objects)
    
    print(Entry_Objects)


    #files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r9364_p3980.bsm4t-21.2.164-1-0-mc16a_output_root/"
    #Entry_Objects = ObjectsFromFile(files, "nominal", ["top_FromRes", "truth_top_pt"])
    #FileObjectsToArrays(Entry_Objects)
