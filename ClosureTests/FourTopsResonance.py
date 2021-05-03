from BaseFunctions.FourTopsResonance import ReadLeafsFromResonance
import matplotlib.pyplot as plt


def PlottingResonance():

    files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    InvTopMass, InvResonanceMass = ReadLeafsFromResonance(files)
   
    #Create a simple sub-plot 
    plt.figure(figsize=(16, 8), dpi=1000) 
    
    plt.subplot(121)
    plt.title("Invariant Top Mass") 
    plt.hist(InvTopMass, align="left", bins=500, range=(172, 173), density=True)
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events")
    
    plt.subplot(122)    
    plt.title("Resonance") 
    plt.hist(InvResonanceMass, align="left", bins=500, range=(500, 1500), density=True)
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events")

    plt.suptitle("Invariant Mass Distribution: Top (Left) Resonance (Right)")
    plt.savefig("./ExamplePlots/Tops_Resonance.png")
    
