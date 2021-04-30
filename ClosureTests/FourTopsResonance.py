from BaseFunctions.FourTopsResonance import ReadLeafsFromResonance


def PlottingResonance():

    files = "/CERN/Grid/Samples/user.tnommens.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r10201_p4174.bsm4t-21.2.102-4-0-mc16d_output_root/user.tnommens.24703615._000001.output.root"
    
    Entry = ReadLeafsFromResonance(files)
