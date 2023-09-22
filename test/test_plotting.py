from AnalysisG._cmodules.cPlots import TH1F
from AnalysisG.Tools import Tools


def test_TH1F():
    x = TH1F()
    x.Title = "test-title"
    x.LaTeX = False
    x.xData = [1, 2, 3, 4]
    x.xMin = 0
    x.xMax = 6
    x.xBins = 6
    x.SaveFigure()
    t = Tools()
    t.rm("Plots")











if __name__ == "__main__":
    test_TH1F()
