import matplotlib.pyplot as plt
from particle import Particle
import os
import numpy as np

def Plotting(Title, Data, bins, ranges = "", xlabels = "", ylabels = ""):
    
    Title_L = type(Title)
    Data_L = type(Data,)
    Bins_L = type(bins,)
    Ranges_L = type(ranges)
    XLables_L = type(xlabels)
    YLables_L = type(ylabels)
    
    if Title_L == list and Data_L == list and Ranges_L == list:
        if len(ranges) == 2 and isinstance(ranges[0], float):
            ranges = [ranges]*len(Title)
        elif len(ranges) == 2 and isinstance(ranges[0], int):
            ranges = [ranges]*len(Title)
        if type(bins) == int:
            bins = [bins]*len(Title)
        if type(xlabels) == str:
            xlabels = [xlabels]*len(Title)
        
        sub_p = len(Title)
        plt.figure(figsize=(sub_p*8, 8), dpi = 500)
        for i in range(sub_p):
            subIndex = int(str(1)+str(sub_p)+str(i+1))
            PlotHist(plt, subIndex, Title[i], Data[i], bins[i], ranges[i], xlabels[i])
    
    return plt

def PlotHist(plt, subIndex, title, data, bins, ranges, xlabels = "", ylabels = ""):
    
    print("Compiling subfig: " + title)
    plt.subplot(subIndex)
    plt.title(title)
    plt.hist(data, align = "left", bins = bins, range=(ranges[0], ranges[1]))
    
    if xlabels != "":
        plt.xlabel(xlabels)
 
    if ylabels != "":
        plt.ylabel(ylabels)


def PlotSpectra(Output, key, Subdir):
    try:
        os.mkdir("./ExamplePlots/"+Subdir+"/") 
    except FileExistsError:
        pass
    for i in Output[key]:
        if str(i) == str(0):
            name = "TruthJet"
        
        else:
            try:
                name = Particle.from_pdgid(i).name
            except ValueError:
                name = i
        plt.figure(figsize=(8,8), dpi=500)
        
        data = Output[key][i]
        min_ = -2
        max_ = round(max(data), 3)

        plt.title("Invariant Mass Spectrum of: " + name + " PDGID: " + str(i))
        plt.hist(data, align = "left", bins = 1000, range=(min_, max_))
        plt.xlabel("Invariant Mass in GeV")
        plt.savefig("./ExamplePlots/"+ Subdir+ "/" + name + ".png")
        plt.close()
        plt.clf()

def PlotADetector(Detector, Truth):
    import numpy as np

    f = plt.figure()
    ax = f.add_subplot(projection="3d")
   
    x_L = []
    y_L = []
    z_L = []
    for i in Detector:
        x_L = 40*np.cos(i.Phi)*np.sin(i.Theta)
        y_L = 40*np.sin(i.Phi)*np.sin(i.Theta)
        z_L = 40*np.cos(i.Theta)

        ax.scatter(x_L, y_L, z_L, marker = "o", cmap = "red")

    for i in Truth:
        x = 20*np.cos(i.Phi)*np.sin(i.Theta)
        y = 20*np.sin(i.Phi)*np.sin(i.Theta)
        z = 20*np.cos(i.Theta)
        ax.plot([0, x], [0, y], [0, z], linestyle="solid", linewidth=1)
        ax.scatter(x, y, z, marker = "^", color="gray")
        
        for c in i.DecayProducts:
            x_c = 10*np.cos(c.Phi)*np.sin(c.Theta) + x
            y_c = 10*np.sin(c.Phi)*np.sin(c.Theta) + y
            z_c = 10*np.cos(c.Theta) + z
            ax.plot([x, x_c], [y, y_c], [z, z_c], linestyle="--", linewidth=1, color = "black")
            ax.scatter(x_c, y_c, z_c, marker="*", color="blue")

            for k in c.DecayProducts:
                x_k = 10*np.cos(k.Phi)*np.sin(c.Theta) + x_c
                y_k = 10*np.sin(k.Phi)*np.sin(c.Theta) + y_c
                z_k = 10*np.cos(k.Theta) + z_c
                ax.plot([x_c, x_k], [y_c, y_k], [z_c, z_k], linestyle="solid", linewidth=1, color = "green")
                ax.scatter(x_k, y_k, z_k, marker="^", color="orange")

          
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()
