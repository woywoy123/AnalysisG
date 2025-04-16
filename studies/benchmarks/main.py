import figures
from loader import construct, assignment

def default(tl):
    tl.Style = "ATLAS"
    tl.DPI = 250
    tl.TitleSize = 25
#    tl.AutoScaling = True
    tl.yScaling = 10*0.75
    tl.xScaling = 15*0.6
    tl.FontSize = 20
    tl.AxisSize = 20
    tl.LegendSize = 14

pth = "/CERN/thesis-data/benchmark/"
devices = ["a100", "h100", "v100", "a30"]

pkt = construct(pth, devices)
pkt = assignment(pkt)

fx = figures
fx.defaults = default
fx.entry(pkt, devices)



