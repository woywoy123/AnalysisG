from AnalysisTopGNN.Plotting import CommonFunctions
import math
import statistics

class Functions(CommonFunctions):
    def __init__(self):
        CommonFunctions.__init__(self)

        # --- Cosmetic --- #
        self.LineStyle = None
        self.Marker = None
 
        # --- Line Properties --- #
        self.DefineAxisData("x")
        self.DefineAxisData("y")
        self.DefineAxisData("up_y", True)
        self.DefineAxisData("down_y", True)
        self.ErrorBars = False
        self.DoStatistics = False
   
    def DefineRange(self, Dims):
        _min  = True if self.Get(Dims + "Min") == None else False
        _max  = True if self.Get(Dims + "Max") == None else False       
        self.DefineCommonRange(Dims)       
        self.Set(Dims + "Min", self.Get(Dims + "Min")*(1 - 0.1*_min))
        self.Set(Dims + "Max", self.Get(Dims + "Max")*(1 + 0.1*_max))

    def ApplyRandomMarker(self, obj):
        ptr = ["x", "o", "O"]
        if obj.Marker == True:
            random.shuffle(ptr)
        elif obj.Marker != None:
            return 
        obj.Marker = ptr[0] 
    
    def MakeStatistics(self, inpt):
        if isinstance(inpt, dict):
            xinpt = list(inpt.keys())
            inpt = list(inpt.values())
        else:
            xinpt = [i+1 for i in range(len(inpt))]            

        output = {}
        for i in range(len(inpt)):
            mean = statistics.mean(inpt[i])
            stdev = statistics.pstdev(inpt[i], mean)/math.sqrt(len(inpt[i]))
            output[xinpt[i]] = [mean, stdev]
        return output

class TLine(Functions):
    
    def __init__(self, **kargs):
        Functions.__init__(self)

        self.ApplyInput(kargs)
        self.Caller = "TLINE"
 
    def Compile(self, Reset = True):
        self.Precompiler()
        if Reset:
            self.ResetPLT()
            self.DefineStyle()
            self.ApplyToPLT()
        self.ApplyRandomMarker(self)

        if self.DoStatistics:
            self._temp = self.xData
            out = self.MakeStatistics(self.xData)
            self.xData = [i for i in out]
            self.yData = [out[i][0] for i in out]
            self.up_yData = [out[i][1] for i in out]
            self.down_yData = [out[i][1] for i in out]

        out = self.SanitizeData(self.xData)
        out += self.SanitizeData(self.yData)
        out = list(set(out))

        er_out = self.SanitizeData(self.down_yData)
        er_out += self.SanitizeData(self.up_yData)
        er_out = list(set(er_out)) 

        self.xData = [self.xData[i] for i in range(len(self.xData)) if i not in out]
        self.yData = [self.yData[i] for i in range(len(self.yData)) if i not in out]      
        self.up_yData = [self.up_yData[i] for i in range(len(self.up_yData)) if i not in out]
        self.down_yData = [self.down_yData[i] for i in range(len(self.down_yData)) if i not in out]

        self.DefineRange("y")
        self.DefineRange("x")
        
        if len(self.xData) != len(self.yData):
            return 

        if len(self.up_yData) != 0 and len(self.down_yData) != 0:
            self.PLT.errorbar(x = self.xData, y = self.yData, yerr = [self.up_yData, self.down_yData], 
                        linestyle = self.LineStyle, color = self.Color, marker = self.Marker, 
                        linewidth = 1, capsize = 3, label = self.Title) 
        else:
            self.PLT.plot(self.xData, self.yData, marker = self.Marker, color = self.Color, linewidth = 1, label = self.Title)
        
        if Reset:
            self.PLT.title(self.Title)
            self.PLT.xlabel(self.xTitle)
            self.PLT.ylabel(self.yTitle)
            self.PLT.xlim(self.xMin, self.xMax)
            self.PLT.ylim(self.yMin, self.yMax)

        if self.Logarithmic:
            self.PLT.yscale("log")
        if self.DoStatistics:
            self.xData = self._temp

class CombineTLine(Functions):

    def __init__(self, **kwargs):
        Functions.__init__(self)

        self.Lines = []
        self.Colors = []
        
        self.ApplyInput(kwargs)
        self.Caller = "COMBINED-TLINE"

        self.ResetPLT()
    
    def ConsistencyCheck(self):
        for i in self.Lines:
            self.yData += i.yData
            self.xData += i.xData
 
        out = self.SanitizeData(self.xData)
        out += self.SanitizeData(self.yData)
        out = list(set(out))
        
        self.xData = [self.xData[i] for i in range(len(self.xData)) if i not in out]
        self.yData = [self.yData[i] for i in range(len(self.yData)) if i not in out]   

        self.DefineRange("x")
        self.DefineRange("y")
        
        for i in self.Lines:
            i.xMin = self.xMin
            i.xMax = self.xMax
            i.yMin = self.yMin
            i.yMax = self.yMax

            if self.xTitle == None:
                self.xTitle = i.xTitle
            if self.yTitle == None:
                self.yTitle = i.yTitle
            
            self.ApplyRandomColor(i)
            self.ApplyRandomMarker(i)
            i.Compile()

    def Compile(self):

        self.ConsistencyCheck()

        self.ResetPLT()
        self.DefineStyle()
        self.ApplyToPLT()
       
        for i in self.Lines:
            i.Compile(False)

        self.PLT.legend()

        if self.Logarithmic:
            self.PLT.yscale("log")

        self.PLT.title(self.Title)
        self.PLT.xlabel(self.xTitle)
        self.PLT.ylabel(self.yTitle)
        self.PLT.xlim(self.xMin, self.xMax)
        self.PLT.ylim(self.yMin, self.yMax)
