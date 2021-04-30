import random
from BaseFunctions.Plotting import HistogramGenerator

def TestSimplePlotting():

    # Generate a random data sample 
    entries = []
    for i in range(100000):
        entries.append(random.random())
    
    HistogramGenerator(entries, 100)
