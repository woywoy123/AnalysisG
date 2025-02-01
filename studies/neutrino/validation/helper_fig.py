from AnalysisG.core.plotting import TH1F, TH2F, TLine

def template_hist(title, xdata, color):
    th1t = TH1F()
    th1t.Title = title
    th1t.Alpha = 0.3
    th1t.xData = xdata
    th1t.Color = color
    return th1t

def LossStatistics():
    a = topchildren_nunu_build()
    b = toptruthjets_nunu_build()
    c = topjetchild_nunu_build()
    d = topdetector_nunu_build()

    a_nevents = len(a["truth_nux"]["tmass"]["n1"])
    a_c1 = a["r1_cu"]["missed"] * 100 / a_nevents
    a_c2 = a["r2_cu"]["missed"] * 100 / a_nevents
    a_r1 = a["r1_rf"]["missed"] * 100 / a_nevents
    a_r2 = a["r2_rf"]["missed"] * 100 / a_nevents

    b_nevents = len(b["truth_nux"]["tmass"]["n1"])
    b_c1 = b["r1_cu"]["missed"] * 100 / b_nevents
    b_c2 = b["r2_cu"]["missed"] * 100 / b_nevents
    b_r1 = b["r1_rf"]["missed"] * 100 / b_nevents
    b_r2 = b["r2_rf"]["missed"] * 100 / b_nevents

    c_nevents = len(c["truth_nux"]["tmass"]["n1"])
    c_c1 = c["r1_cu"]["missed"] * 100 / c_nevents
    c_c2 = c["r2_cu"]["missed"] * 100 / c_nevents
    c_r1 = c["r1_rf"]["missed"] * 100 / c_nevents
    c_r2 = c["r2_rf"]["missed"] * 100 / c_nevents

    d_nevents = len(d["truth_nux"]["tmass"]["n1"])
    d_c1 = d["r1_cu"]["missed"] * 100 / d_nevents
    d_c2 = d["r2_cu"]["missed"] * 100 / d_nevents
    d_r1 = d["r1_rf"]["missed"] * 100 / d_nevents
    d_r2 = d["r2_rf"]["missed"] * 100 / d_nevents

    sx = 3
    print("------------ Truth Children ------------")
    print("raw events:", a_nevents)
    print("cuda - dyn:", round(a_c1, sx), "cuda-static:", round(a_c2, sx))
    print("ref - dyn:" , round(a_r1, sx), "ref-static:" , round(a_r2, sx))

    print("------------ Truth Jets + Truth Children ------------")
    print("raw events:", b_nevents)
    print("cuda - dyn:", round(b_c1, sx), "cuda-static:", round(b_c2, sx))
    print("ref - dyn:" , round(b_r1, sx), "ref-static:" , round(b_r2, sx))

    print("------------ Jets + Truth Children ------------")
    print("raw events:", c_nevents)
    print("cuda - dyn:", round(c_c1, sx), "cuda-static:", round(c_c2, sx))
    print("ref - dyn:" , round(c_r1, sx), "ref-static:" , round(c_r2, sx))

    print("------------ Jets + Leptons ------------")
    print("raw events:", d_nevents)
    print("cuda - dyn:", round(d_c1, sx), "cuda-static:", round(d_c2, sx))
    print("ref - dyn:" , round(d_r1, sx), "ref-static:" , round(d_r2, sx))


