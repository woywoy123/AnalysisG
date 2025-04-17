from AnalysisG.core.plotting import TLine
global defaults 

def colr(dev):
    cols = {"a100" : "red", "h100" : "blue", "v100" : "green", "a30" : "orange"}
    return cols[dev]

def fancy(dev):
    tlt = {"a100" : "Ampere A100", "h100" : "Hopper H100", "v100" : "Tesla V100", "a30" : "Ampere A30"}
    return "\\texttt{" + tlt[dev] + "}"

def fancy_title(key):
    maps = {
            "m"           : "Invariant Mass of Four Vector ($M$)", 
            "p"           : "Magnitude of Momentum Vector ($\\sqrt{\\mathbf{p} \\cdot \\mathbf{p}}$)", 
            "mt"          : "Transverse Invariant Mass ($M_T$)", 
            "beta"        : "Fractional Velocity Relative to Speed of Light ($\\beta$)", 
            "m2"          : "Square Invariant Mass of Four Vector ($M^2$)", 
            "p2"          : "Magnitude of Momentum Vector ($\\mathbf{p} \\cdot \\mathbf{p}$)", 
            "mt2"         : "Transverse Invariant Mass Squared ($M^{2}_T$)", 
            "beta2"       : "Fractional Velocity Relative to Speed of Light Squared ($\\beta^2$)", 
            "theta"       : "$\\cos^{-1}(P_z / \\sqrt{\\mathbf{p} \\cdot \\mathbf{p}})$", 
            "deltaR"      : "$\\Delta R$ Between Two Four Vectors",
            "dot"         : "The Dot Product of Matrices A and B ($A \\cdot B$)", 
            "cofactor"    : "Cofactors for a $3 \\times 3$ Matrix", 
            "determinant" : "Determinant for a $3 \\times 3$ Matrix ($|A|$)",
            "eigenvalue"  : "Eigenvalues for a Symmetric $3 \\times 3$ Matrix ($(\\mathbf{A} - \\lambda \\mathbf{I})\\mathbf{v}$)", 
            "inverse"     : "Inverse of a $3 \\times 3$ Matrix ($A^{-1}$)",
            "px"          : "Polar ($p_T, \\eta, \\phi$) Vector to Cartesian $x$-Momentum", 
            "py"          : "Polar ($p_T, \\eta, \\phi$) Vector to Cartesian $y$-Momentum", 
            "pz"          : "Polar ($p_T, \\eta, \\phi$) Vector to Cartesian $z$-Momentum", 
            "eta"         : "Cartesian ($p_x, p_y, p_z$) Vector to Rapidity Polar Component ($\\eta$)", 
            "phi"         : "Cartesian ($p_x, p_y, p_z$) Vector to Azimuthal Polar Component ($\\phi$)", 
            "pt"          : "Cartesian ($p_x, p_y, p_z$) Vector to Transverse Momentum Polar Component ($p_T$)", 
            "pxpypz"      : "Polar ($p_T, \\eta, \\phi$) to Cartesian Vector ($p_x, p_y, p_z$)", 
            "pxpypze"     : "Polar ($p_T, \\eta, \\phi, E$) to Cartesian Vector ($p_x, p_y, p_z, E$)", 
            "ptetaphi"    : "Cartesian ($p_x, p_y, p_z$) to Polar Vector ($p_T, \\eta, \\phi$)",
            "ptetaphie"   : "Cartesian ($p_x, p_y, p_z, E$) to Polar Vector ($p_T, \\eta, \\phi, E$)",
            "basematrix"  : "H-Matrix (Neutrino Extended Matrix Representation)",
    }
    return maps[key]

def fancy_modes():
    modx = {
            "cpu"        : "Time (CPU) - Lower is Better", 
            "cuda_t"     : "Time (CUDA - Tensor) - Lower is Better", 
            "cuda_k"     : "Time (CUDA - Kernel) - Lower is Better", 
            "cpu_cuda_t" : "Ratio (CPU / CUDA - Tensor) - Higher is Better", 
            "cpu_cuda_k" : "Ratio (CPU / CUDA - Kernel) - Higher is Better", 
            "cuda_t_k"   : "Ratio CUDA (Tensor / Kernel) - Higher is Better"
    }
    return modx

def fnames(key):
    modx = {
            "cpu"        : "cpu_time", 
            "cuda_t"     : "cuda_tensor_time", 
            "cuda_k"     : "cuda_kernel_time", 
            "cpu_cuda_t" : "ratio_cpu_cuda_tensor",
            "cpu_cuda_k" : "ratio_cpu_cuda_kernel",
            "cuda_t_k"   : "ratio_cuda_torch_kernel" 
    }
    return modx[key]


def xdata(data, mode): return [getattr(i, mode).idx / 1000 for i in data][1:]
def ydata(data, mode): return [getattr(i, mode).mu  for i in data][1:]
def edata(data, mode): return [getattr(i, mode).sig * (1. / (10000.-1.)**0.5) for i in data][1:]

def template_line(title, data, mode, linst = "-", misc = None):
    tl = TLine()
    tl.Title = fancy(title) if misc is None else fancy(title) + " ($\\texttt{" + misc + "}$)"
    tl.LineWidth = 1.0
    tl.CapSize = 2.0
    tl.ErrorBars = True
    tl.xData = xdata(data, mode)
    tl.yData = ydata(data, mode) 
    tl.yDataUp   = edata(data, mode)
    tl.yDataDown = edata(data, mode)
    tl.Color = colr(title)
    tl.LineStyle = linst
    return tl

def template_Tline(ytitle, opx, fname, lines):
    th = TLine()
    defaults(th)

    th.Lines = lines
    th.xMin = 0
    th.xMax = 1000
    th.yMin = 0
    th.xStep = 100
    th.yTitle = ytitle
    th.OutputDirectory = "./Figures/" + opx
    th.xTitle = "Length of Tensor - (Units of 1000)"
    th.Filename = fname
    if "time" in fname: return th

    m_ = 0
    for i in lines: m_ = max(i.yData) if m_ < max(i.yData) else m_
    
    if   m_ < 2:    th.yStep = 0.2
    elif m_ < 3:    th.yStep = 0.2
    elif m_ < 5:    th.yStep = 0.5
    elif m_ < 10:   th.yStep = 0.5
    elif m_ < 20:   th.yStep = 1
    elif m_ < 30:   th.yStep = 2
    elif m_ < 40:   th.yStep = 4
    elif m_ < 100:  th.yStep = 10
    else: th.yLogarithmic = True; th.yMin = 1.0

    if   m_ < 2:    th.yMax = 2.1
    elif m_ < 3:    th.yMax = 3.1
    elif m_ < 5:    th.yMax = 5.1
    elif m_ < 10:   th.yMax = 10.1
    elif m_ < 20:   th.yMax = 20.1
    elif m_ < 30:   th.yMax = 30.1
    elif m_ < 40:   th.yMax = 40.1
    elif m_ < 100:  th.yMax = 100.1
    return th

def build_operators(pkt):
    modx = fancy_modes()
    for opx in pkt:
        for md in modx:
            lines = {}
            for dev in pkt[opx]: lines[dev] = template_line(dev, pkt[opx][dev], md)
            tl = template_Tline(modx[md], opx, fnames(md), list(lines.values()))
            if opx == "eigenvalue": tl.xMax = 600; tl.yLogarithmic = False
            tl.Title = fancy_title(opx)
            tl.SaveFigure()

def build_neutrino(pkt):
    modx = fancy_modes()
    for opx in pkt:
        for md in modx:
            lines = {}
            for dev in pkt[opx]: lines[dev] = template_line(dev, pkt[opx][dev], md)
            tl = template_Tline(modx[md], opx, fnames(md), list(lines.values()))
            tl.Title = fancy_title(opx)
            tl.SaveFigure()

def build_physics(pkt, frame):
    modx = fancy_modes()
    for opx in pkt:
        for md in modx:
            lines = []
            for tnx in pkt[opx]:
                hx = "solid" if tnx == "separate" else "dotted"
                rn = "Separate" if tnx == "separate" else "Combined"
                for dev in pkt[opx][tnx]: lines.append(template_line(dev, pkt[opx][tnx][dev], md, hx, rn))
            tl = template_Tline(modx[md], opx, fnames(md) + "_" + frame.lower(), lines)
            tl.Title = fancy_title(opx) + " (From " + frame + ")"
            tl.SaveFigure()

def build_transform(pkt):
    modx = fancy_modes()
    for opx in pkt:
        for md in modx:
            lines = []
            for tnx in pkt[opx]:
                hx = "solid" if tnx == "separate" else "dotted"
                rn = "Separate" if tnx == "separate" else "Combined"
                for dev in pkt[opx][tnx]: lines.append(template_line(dev, pkt[opx][tnx][dev], md, hx, rn))
            tl = template_Tline(modx[md], opx, fnames(md), lines)
            tl.Title = fancy_title(opx)
            tl.SaveFigure()

def entry(pkt, devices):
    for i in pkt:
        if i == "operators"          : build_operators(pkt[i]); continue
        if i == "neutrino"           : build_neutrino(pkt[i]);  continue
        if i == "physics-polar"      : build_physics(pkt[i], "Polar");   continue
        if i == "physics-cartesian"  : build_physics(pkt[i], "Cartesian");   continue
        if i == "transform-polar"    : build_transform(pkt[i]); continue
        if i == "transform-cartesian": build_transform(pkt[i]); continue
