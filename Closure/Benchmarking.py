from PathNetOptimizer_cpp import PathCombination
from itertools import combinations
import torch 
import time
from Functions.Plotting.Histograms import CombineTGraph,TGraph

def Combinatorials():
   

    node = 20

    combi = TGraph()
    combi.Title = "Combinatorials as a function of Nodes"
    combi.xTitle = "Number of Nodes"
    combi.Filename = "Combinatorials.png"
    combi.yTitle = "Number of Combinations"
    combi.xMax = node+1

    pyc = TGraph()
    pyc.Title = "Python"
    pyc.xTitle = "Number of Nodes"
    pyc.yTitle = "Time (s)"
    pyc.xMax = node+1

    cusx = TGraph()
    cusx.Title = "C++"
    cusx.xTitle = "Number of Nodes"
    cusx.yTitle = "Time (s)"
    cusx.xMax = node+1
    
    for p in range(2, node+1):
        index = torch.tensor([[ i , j] for i in range(p) for j in range(p) if i != j ]).t()
        Adj = torch.zeros(p, p)
        Adj[index[0], index[1]] = 1
        unique = torch.unique(index).tolist()

        time_s = time.time()
        store = []
        for i in range(1, p):
            x = torch.tensor(list(combinations(unique, r = i+1)), device = "cuda")
            store.append(x)
        pyc.yData.append(time.time() - time_s)
        pyc.xData.append(p)
        
        time_s = time.time()
        Combi = PathCombination(p, p, "cuda")
        cusx.yData.append(time.time() - time_s)
        cusx.xData.append(p)

        combi.xData.append(p)
        combi.yData.append(len(Combi))
        print(len(Combi))
        del Combi

    pyc.Color = "blue"
    cusx.Color = "green"

    pyc.Save("Plots/Combinatorials")
    cusx.Save("Plots/Combinatorials")
    combi.Save("Plots/Combinatorials")
    
    com = CombineTGraph()
    com.Title = "Comparison of Different Combinatorial Generators"
    com.Lines = [cusx, pyc]
    com.yMax = 15
    com.Filename = "Combinatorials.png"
    com.Save("Plots/Combinatorials")


    exit()
    return True
