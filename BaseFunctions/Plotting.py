import matplotlib.pyplot as plt

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
            PlotHist(plt, subIndex, Title[i], Data[i], bins[i], ranges[i], xlabels)
    
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


