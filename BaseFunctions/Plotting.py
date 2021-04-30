import matplotlib.pyplot as plt


def HistogramGenerator(data, bins = 10):
    
    plt.hist(data, bins)
    plt.show() 
    
