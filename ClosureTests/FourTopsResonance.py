from BaseFunctions.FourTopsResonance import *
import matplotlib.pyplot as plt
from particle import Particle
import numpy as np

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
    plt.figure(figsize=(32, 8), dpi=500) 
    
    plt.subplot(141)
    plt.title("Invariant Child Mass") 
    plt.ylim(0, 0.2)
    plt.hist(Output["SGDaughterM"], align="left", bins=10000, range=(-200, 10000), density=True)
    plt.xlabel("Invariant Mass (MeV)")
    plt.ylabel("Events")
     
    plt.subplot(142)    
    plt.title("Resonance From Children") 
    plt.hist(Output["SGMass"], align="left", bins= 300, range=(0, 1500), density=True)
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events")
   
    plt.subplot(143)    
    plt.title("PID of Particles Contributing to Resonance") 
    plt.hist(Output["SGDaughterPDG"], align="left", bins=40, range=(-20, 20), density=True)
    plt.xlabel("PID of Particles")
    plt.ylabel("Events")

    plt.subplot(144)    
    plt.title("Mass Spectrum of Top Particle (From Children)") 
    plt.hist(Output["TopMass"], align="left", bins=200, range=(0, 200), density=True)
    plt.xlabel("Mass in GeV")
    plt.ylabel("Events")
        
    plt.savefig("./ExamplePlots/Tops_Resonance_FromChildren.png")
    plt.close()
    plt.clf()
    
    #Create a simple sub-plot 
    plt.figure(figsize=(32, 8), dpi=500) 
    
    plt.subplot(141)
    plt.title("Invariant Child Mass (Init Branch)") 
    plt.ylim(0, 0.2)
    plt.hist(Output["SGDaughterM_init"], align="left", bins=10000, range=(-200, 10000), density=True)
    plt.xlabel("Invariant Mass (MeV)")
    plt.ylabel("Events")
    
    plt.subplot(142)    
    plt.title("Resonance From Children (Init Branch)") 
    plt.hist(Output["SGMass_init"], align="left", bins= 300, range=(0, 1500), density=True)
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events")
   
    plt.subplot(143)    
    plt.title("PID of Particles Contributing to Resonance (Init Branch)") 
    plt.hist(Output["SGDaughterPDG_init"], align="left", bins=40, range=(-20, 20), density=True)
    plt.xlabel("PID of Particles")
    plt.ylabel("Events")

    plt.subplot(144)    
    plt.title("Mass Spectrum of Top Particle (From Children) (Init Branch)") 
    plt.hist(Output["TopMass_init"], align="left", bins=200, range=(0, 200), density=True)
    plt.xlabel("Mass in GeV")
    plt.ylabel("Events")
        
    plt.savefig("./ExamplePlots/Tops_Resonance_FromChildren_init.png")
    plt.close()
    plt.clf()

    
    #Create a simple sub-plot 
    plt.figure(figsize=(32, 8), dpi=500) 
    
    plt.subplot(131)
    plt.title("Invariant Child Mass of Spectator Tops") 
    plt.ylim(0, 0.2)
    plt.hist(Output["SpecDaughterM"], align="left", bins=10000, range=(-200, 10000), density=True)
    plt.xlabel("Invariant Mass (MeV)")
    plt.ylabel("Events")
    
    plt.subplot(132)    
    plt.title("PID of Child Particles of Spectators") 
    plt.hist(Output["SpecDaughterPDG"], align="left", bins=40, range=(-20, 20), density=True)
    plt.xlabel("PID of Particles")
    plt.ylabel("Events")

    plt.subplot(133)    
    plt.title("Mass Spectrum of Top Particle (From Children)") 
    plt.hist(Output["SpecTopMass"], align="left", bins=200, range=(0, 200), density=True)
    plt.xlabel("Mass in GeV")
    plt.ylabel("Events")
        
    plt.savefig("./ExamplePlots/Spectator_Tops_FromChildren.png")
    plt.close()
    plt.clf()
    
    #Create a simple sub-plot 
    plt.figure(figsize=(32, 8), dpi=500) 
    
    plt.subplot(131)
    plt.title("Invariant Child Mass of Spectator Tops (Init Branch)") 
    plt.ylim(0, 0.2)
    plt.hist(Output["SpecDaughterM_init"], align="left", bins=10000, range=(-200, 10000), density=True)
    plt.xlabel("Invariant Mass (MeV)")
    plt.ylabel("Events")
    
    plt.subplot(132)    
    plt.title("PID of Child Particles of Spectators (Init Branch)") 
    plt.hist(Output["SpecDaughterPDG_init"], align="left", bins=40, range=(-20, 20), density=True)
    plt.xlabel("PID of Particles")
    plt.ylabel("Events")

    plt.subplot(133)    
    plt.title("Mass Spectrum of Top Particle (From Children) (Init Branch)") 
    plt.hist(Output["SpecTopMass_init"], align="left", bins=200, range=(0, 200), density=True)
    plt.xlabel("Mass in GeV")
    plt.ylabel("Events")
        
    plt.savefig("./ExamplePlots/Spectator_Tops_FromChildren_Init.png")
    plt.close()
    plt.clf()
    
    def PlotSpectra(Output, key, Subdir):
        for i in Output[key]:
            plt.figure(figsize=(8,8), dpi=500)
            name = Particle.from_pdgid(i).name
            print(name) 
            data = np.asarray(Output[key][i])
            min_ = data.min(axis =0)
            max_ = data.max(axis =0)
            plt.title("Invariant Mass Spectrum of: " + name + " PDGID: " + str(i))
            plt.hist(data, align = "left", bins = int(max_ - min_), range=(min_, max_), density=True)
            plt.xlabel("Invariant Mass in MeV")
            plt.savefig("./ExamplePlots/"+ Subdir+ "/" + name + ".png")
            plt.close()
            plt.clf()


    
    PlotSpectra(Output, "SGDMassPDG", "MassSpec")
    PlotSpectra(Output, "SGDMassPDG_init", "MassSpec_init")
    PlotSpectra(Output, "SpecMassPDG", "MassSpectator")
    PlotSpectra(Output, "SpecMassPDG_init", "MassSpectator_init")


def TestChildToTruthJet():
    files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    ChildToTruthJet(files)

