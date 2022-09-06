from AnalysisTopGNN.Plotting import Settings, CommonFunctions
from AnalysisTopGNN.IO import WriteDirectory
from AnalysisTopGNN.Tools import Notification


class CommonFunctions(CommonFunctions, Settings, WriteDirectory):
    def __init__(self):
        Notification.__init__(self)
        WriteDirectory.__init__(self)

        # --- Element Sizes ---
        self.FontSize = 10
        self.LabelSize = 12.5
        self.TitleSize = 10
        self.LegendSize = 10

        # --- Figure Settings --- #
        self.Scaling = 1.25
        self.DPI = 250

        # --- Cosmetic --- #
        self.Style = None
        self.ATLASData = False
        self.ATLASYear = None
        self.ATLASLumi = None
        self.ATLASCom = None
        self.LineStyle = None
        self.Color = None
        self.Marker = None
 
        # --- Line Properties --- #
        self.Title = None
        self.DefineAxis("x")
        self.DefineAxis("y")
        self.DefineAxis("up_y")
        self.DefineAxis("down_y")
        self.ErrorBars = False

        self.OutputDirectory = "Plots"
        self.Filename = None

   
    def DefineAxis(self, Dim):
        self.Set(Dim + "Min", None)
        self.Set(Dim + "Max", None)
        self.Set(Dim + "Title", None)
        self.Set(Dim + "Data", [])   

    def DefineRange(self, Dims):
        if self.Get(Dims + "Min") == None:
            self.Set(Dims + "Min", min(self.Get(Dims + "Data")))

        if self.Get(Dims + "Max") == None:
            self.Set(Dims + "Max", max(self.Get(Dims + "Data")))

    def ApplyRandomTexture(self, obj):
        ptr = ["-" , "+" , "x", "o", "O", ".", "*" ]
        if obj.LineStyle == True:
            random.shuffle(ptr)
        elif obj.LineStyle != None:
            return 
        obj.LineStyle = ptr[0] 

    def ApplyRandomMarker(self, obj):
        ptr = ["x", "o", "O"]
        if obj.Marker == True:
            random.shuffle(ptr)
        elif obj.Marker != None:
            return 
        obj.Marker = ptr[0] 

    def GetBinWidth(self):
        pass
    
    def CenteringBins(self):
        pass


class TLine(CommonFunctions):
    
    def __init__(self, **kargs):
        CommonFunctions.__init__(self)

        self.ApplyInput(kargs)
        self.Caller = "TLINE"


 
    def Compile(self):
        self.ResetPLT()
        self.DefineStyle()
        self.ApplyToPLT()


        self.ApplyRandomColor(self)
        self.ApplyRandomColor(self)


        if len(self.up_yData) != 0:
            self.PLT.errorbar(x = self.xData, y = self.yData, yerr = [self.up_yData, self.down_yData], 
                        linestyle = self.LineStyle, color = self.Color, marker = self.Marker, 
                        linewidth = 1, capsize = 3) 
