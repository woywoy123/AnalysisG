from BaseFunctions.FourTopsResonance import *
import matplotlib.pyplot as plt


def TestResonanceFromTruthTops():

    files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    InvTopMass, InvResonanceMass, SpectatorMass = ResonanceFromTruthTops(files)
  
    #Create a simple sub-plot 
    plt.figure(figsize=(16, 8), dpi=1000) 
    
    plt.subplot(131)
    plt.title("Invariant Top Mass") 
    plt.hist(InvTopMass, align="left", bins=500, range=(172, 173), density=True)
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events")
    
    plt.subplot(132)    
    plt.title("Resonance") 
    plt.hist(InvResonanceMass, align="left", bins=1000, range=(500, 1500), density=True)
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events")

    plt.subplot(133)    
    plt.title("Spectator") 
    plt.hist(SpectatorMass, align="left", bins=1500, range=(0, 1500), density=True)
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events")

    plt.suptitle("Invariant Mass Distribution: Top (Left) Resonance (Middle) Spectator (Right)")
    plt.savefig("./ExamplePlots/Tops_Resonance.png")

def TestSignalTopsFromChildren():

    files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    #files = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root" 
    Output = SignalTopsFromChildren(files)
   
    print("Entering the Matplot stuff")

    #Create a simple sub-plot 
    plt.figure(figsize=(32, 8), dpi=100) 
    
    plt.subplot(141)
    plt.title("Invariant Child Mass") 
    plt.hist(Output["SGDaughterM"], align="left", bins=1300, range=(0, 4000), density=True)
    plt.hist(Output["SGDaughterM_init"], align="left", bins=1300, range=(0, 4000), density=True)
    plt.xlabel("Invariant Mass (MeV)")
    plt.ylabel("Events")
    
    plt.subplot(142)    
    plt.title("Resonance From Children") 
    plt.hist(Output["SGMass"], align="left", bins= 1300, range=(200, 1500), density=True)
    plt.hist(Output["SGMass_init"], align="left", bins= 1300, range=(200, 1500), density=True)
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events")

    plt.subplot(143)    
    plt.title("PID of Particles Contributing to Resonance") 
    plt.hist(Output["SGDaughterPDG"], align="left", bins=40, range=(-20, 20), density=True)
    plt.hist(Output["SGDaughterPDG_init"], align="left", bins=40, range=(-20, 20), density=True)
    plt.xlabel("PID of Particles")
    plt.ylabel("Events")

    plt.subplot(144)    
    plt.title("Mass Spectrum of Top Particle (From Children)") 
    plt.hist(Output["TopMass"], align="left", bins=20, range=(100, 200), density=True)
    plt.hist(Output["TopMass_init"], align="left", bins=20, range=(100, 200), density=True)
    plt.xlabel("Mass in GeV")
    plt.ylabel("Events")

    plt.savefig("./ExamplePlots/Tops_Resonance_FromChildren.png")




