from BaseFunctions.FourTopsResonance import *
from BaseFunctions.Plotting import *
import matplotlib.pyplot as plt
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
    plt.savefig("./ExamplePlots/TestResonanceFromTruthTops/Tops_Resonance.png")

def TestSignalTopsFromChildren():

    files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    files = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root" 
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
        
    plt.savefig("./ExamplePlots/TestSignalTopsFromChildren/Tops_Resonance_FromChildren.png")
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
        
    plt.savefig("./ExamplePlots/TestSignalTopsFromChildren/Tops_Resonance_FromChildren_init.png")
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
        
    plt.savefig("./ExamplePlots/TestSignalTopsFromChildren/Spectator_Tops_FromChildren.png")
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
        
    plt.savefig("./ExamplePlots/TestSignalTopsFromChildren/Spectator_Tops_FromChildren_Init.png")
    plt.close()
    plt.clf()
    
    PlotSpectra(Output, "SGDMassPDG", "TestSignalTopsFromChildren/MassSpec")
    PlotSpectra(Output, "SGDMassPDG_init", "TestSignalTopsFromChildren/MassSpec_init")
    PlotSpectra(Output, "SpecMassPDG", "TestSignalTopsFromChildren/MassSpectator")
    PlotSpectra(Output, "SpecMassPDG_init", "TestSignalTopsFromChildren/MassSpectator_init")


def TestChildToTruthJet():
    files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    Output = ChildToTruthJet(files)

    # Plot the Mass spectra of the Resonance using: 1.Truth, 2. Truth Children (init) 3. Children of Children
    Title = ["Truth Tops Branches", "Resonance From Children (init)", "Resonance From Truth Jets"]
    Data = [Output["Mass_Resonance_Truth"], Output["Mass_Resonance_Child"], Output["Mass_Resonance_Child_of_Child"]]
    Bins = 1100
    Ranges = [300, 1400]
    X_Labels = "Invariant Mass (GeV)"
    
    plt = Plotting(Title, Data, Bins, Ranges, X_Labels)
    plt.savefig("./ExamplePlots/TestChildToTruthJet/Resonance.png")
    plt.clf()
    
    # Plot the mass spectrum of tops originating from Signal Tops
    Title = ["Mass of Tops from Signal", "Mass of Signal Tops from (init) Children", "Mass of Signal Tops from Truth Jets"]
    Data = [Output["Mass_Signal_Tops_Truth"], Output["Mass_Signal_Tops_Child"], Output["Mass_Signal_Tops_Child_of_Child"]]
    Bins = 130
    Ranges = [120, 240]
    X_Lables = "Invariant Mass (GeV)"

    plt = Plotting(Title, Data, Bins, Ranges, X_Labels)
    plt.savefig("./ExamplePlots/TestChildToTruthJet/Signal_Tops.png")
    plt.clf()

    # Plot the mass spectrum of tops originating from Spectator Tops
    Title = ["Mass of Tops from Spectator", "Mass of Spectator Tops from (init) Children", "Mass of Spectator Tops from Truth Jets"]
    Data = [Output["Mass_Spectator_Tops_Truth"], Output["Mass_Spectator_Tops_Child"], Output["Mass_Spectator_Tops_Child_of_Child"]]
    Bins = 130
    Ranges = [120, 240]
    X_Lables = "Invariant Mass (GeV)"

    plt = Plotting(Title, Data, Bins, Ranges, X_Labels)
    plt.savefig("./ExamplePlots/TestChildToTruthJet/Spectator_Tops.png")
    plt.clf()

    # Plot the mass spectrum of Children (init) from Signal/Spectator tops
    Title = ["Mass Spectrum from Children of Children from Spectator Tops"]
    Data = [Output["C_C_Mass_Spectator_Child"]]
    Bins = 3000
    Ranges = [-10, 20]
    X_Lables = "Invariant Mass (GeV)"

    plt = Plotting(Title, Data, Bins, Ranges, X_Labels)
    plt.savefig("./ExamplePlots/TestChildToTruthJet/ChildrenOfChildren_Spectator_Tops.png")
    plt.clf()

    Title = ["Mass Spectrum from Children of Children from Signal Tops"]
    Data = [Output["C_C_Mass_Signal_Child"]]
    Bins = 6000
    Ranges = [-1, 50]
    X_Lables = "Invariant Mass (GeV)"

    plt = Plotting(Title, Data, Bins, Ranges, X_Labels)
    plt.savefig("./ExamplePlots/TestChildToTruthJet/ChildrenOfChildren_Signal_Tops.png")
    plt.clf()

    # Individual spectra
    PlotSpectra(Output, "C_Mass_Signal_Child", "TestChildToTruthJet/Signal_MassOfChildren_init")
    PlotSpectra(Output, "C_Mass_Spectator_Child", "TestChildToTruthJet/Spectator_MassOfChildren_init")

    PlotSpectra(Output, "Mass_Signal_Child_Child", "TestChildToTruthJet/Signal_MassTruthJetParticles")
    PlotSpectra(Output, "Mass_Spectator_Child_Child", "TestChildToTruthJet/Spectator_MassTruthJetParticles")


def TestChildToDetectorParticles():

    files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    Output = ChildToDetectorParticles(files)




