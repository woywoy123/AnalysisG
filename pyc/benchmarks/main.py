import sys
sys.path.append("../build/pyc")

from common import *
import fx
import pathlib
import pickle
global dev_name
global pyc

def serialize(data, name, i, pth = "./data"):
    path = pth + "/" + name + "/data-" + str(i) + ".pkl"
    try: return pickle.load(open(path, "rb"))
    except FileNotFoundError: pass
    except: pass
    if data is None: return
    pathlib.Path(pth + "/" + name).mkdir(parents = True, exist_ok = True)
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()

def benchmark(gen, fx, dx, lf, title, iters = 1000, gen_data_only = False, plot_only = False):
    df = {}
    name = gen.__name__
    data, data_ = None, None
    m_ = 0
    dex = ["v100", "a100"]
    for k in dex:
        df[k] = {}
        dev_name = "./" + k
        for i in range(lf):
            lx = dx + i*dx
            data_ = serialize(None, name, lx)
            if not plot_only:
                if data_ is None:
                    data_ = gen(dx, data_, None)
                    serialize(data_, name, lx)
                    print("(Generating) -> ", name, i, lf, lx)
                data = gen(dx, data, data_)
                print("(Loading) -> ", name, i, lf, lx)
                if gen_data_only: continue

            ld = serialize(None, fx.__name__, lx, dev_name)
            if ld is not None: df[k] = ld
            elif plot_only and ld is not None: df[k] = ld
            else:
                try: df[k] = merge(df[k], repeat(fx, data, iters, lx))
                except: break
            if ld is None: serialize(df, fx.__name__, lx, dev_name)
        #my = max(df[k]["cpu/cuda(t)"] +  df[k]["cpu/cuda(k)"])
        #if my > m_: m_ = my

    if dx == lx: return
    if gen_data_only: return
    cols = iter(["red", "blue", "green"])
    ln = sum([MakeLines(df[k], k, next(cols)) for k in dex], [])

    tl = Line(title)
    tl.Lines = ln #MakeLines(df[k])
    tl.xMin = 0
    tl.xMax = lf
    tl.xStep = 50
    tl.yMin = 0
    tl.yMax = 400 #m_*1.2
    tl.DPI = 100
#    tl.yLogarithmic = True
#    tl.yStep = 10
    tl.Style = "ATLAS"
#    tl.ErrorBars = True
    tl.Filename = fx.__name__
    tl.SaveFigure()

def create_polar_combined(dx, dx_old, mrg):
    if dx_old is None: return create_particle(dx)["polar"]
    data = create_particle(dx)["polar"] if mrg is None else mrg
    return torch.cat([data, dx_old], dim = 0)

def create_polar_separate(dx, dx_old, mrg):
    data = create_particle(dx)["polar"] if mrg is None else torch.cat(mrg, -1)
    pt , eta , phi , e  = [data[:, i].view(-1, 1) for i in range(4)]
    if dx_old is None: return [pt, eta, phi, e]

    pt , eta , phi , e  = [data[:, i].view(-1, 1) for i in range(4)]
    pt_, eta_, phi_, e_ = [dx_old[i].view(-1, 1) for i in range(4)]
    return [
            torch.cat([pt , pt_ ], dim = 0),
            torch.cat([eta, eta_], dim = 0),
            torch.cat([phi, phi_], dim = 0),
            torch.cat([e  , e_  ], dim = 0)
    ]

def create_cartesian_combined(dx, dx_old, mrg):
    if dx_old is None: return create_particle(dx)["cartesian"]
    data = create_particle(dx)["cartesian"] if mrg is None else mrg
    return torch.cat([data, dx_old], dim = 0)


def create_cartesian_separate(dx, dx_old, mrg):
    data = create_particle(dx)["cartesian"] if mrg is None else torch.cat(mrg, -1)
    px , py , pz , e  = [data[:, i].view(-1, 1) for i in range(4)]
    if dx_old is None: return [px , py , pz , e]

    px , py , pz , e  = [data[:, i].view(-1, 1) for i in range(4)]
    px_, py_, pz_, e_ = [dx_old[i].view(-1, 1) for i in range(4)]
    return [
            torch.cat([px, px_], dim = 0),
            torch.cat([py, py_], dim = 0),
            torch.cat([pz, pz_], dim = 0),
            torch.cat([e , e_ ], dim = 0)
    ]

def create_matrix_4x4(dx, dx_old, mrg):
    if dx_old is None: return torch.randn((dx, 4, 4), dtype = torch.float64)
    data = torch.randn((dx, 4, 4), dtype = torch.float64) if mrg is None else mrg
    return torch.cat([data, dx_old], dim = 0)

def create_matrix_3x3(dx, dx_old, mrg):
    if dx_old is None: return torch.randn((dx, 3, 3), dtype = torch.float64)
    data = torch.randn((dx, 3, 3), dtype = torch.float64) if mrg is None else mrg
    return torch.cat([data, dx_old], dim = 0)

def create_symmetric_3x3(dx, dx_old, mrg):
    if dx_old is None:
        tx = torch.randn((dx, 3, 3), dtype = torch.float64)
        tx += tx.transpose(1, 2).clone()
        return tx;

    if mrg is None:
        data = torch.randn((dx, 3, 3), dtype = torch.float64)
        data += data.transpose(1, 2).clone()
    else: data = mrg
    return torch.cat([data, dx_old], dim = 0)

def start(mode):
    msg = "" #Performance Ratio Between Reference PyToch and Kernel Operator: \n "
    step_size = 1000
    sampling = 10000
    leng = 1000
    gen_data_only = False
    plot_only = True
    fx.pyx = pyc
    mode = int(mode)

    # ------------------ Transformation --------------------------- #
    #if mode ==  0 or plot_only: benchmark(create_polar_separate, fx.px_separate     , step_size, leng, msg + "(Separate) Polar to Px", sampling, gen_data_only, plot_only)
    #if mode ==  1 or plot_only: benchmark(create_polar_separate, fx.py_separate     , step_size, leng, msg + "(Separate) Polar to Py", sampling, gen_data_only, plot_only)
    #if mode ==  2 or plot_only: benchmark(create_polar_separate, fx.pz_separate     , step_size, leng, msg + "(Separate) Polar to Pz", sampling, gen_data_only, plot_only)
    #if mode ==  3 or plot_only: benchmark(create_polar_separate, fx.pxpypz_separate , step_size, leng, msg + "(Separate) Polar to PxPyPz", sampling, gen_data_only, plot_only)
    #if mode ==  4 or plot_only: benchmark(create_polar_separate, fx.pxpypze_separate, step_size, leng, msg + "(Separate) Polar to PxPyPzE", sampling, gen_data_only, plot_only)
    #if mode ==  5 or plot_only: benchmark(create_cartesian_separate, fx.pt_separate       , step_size, leng, msg + "(Separate) Cartesian to $p_T$", sampling, gen_data_only, plot_only)
    #if mode ==  6 or plot_only: benchmark(create_cartesian_separate, fx.eta_separate      , step_size, leng, msg + "(Separate) Cartesian to $\\eta$", sampling, gen_data_only, plot_only)
    #if mode ==  7 or plot_only: benchmark(create_cartesian_separate, fx.phi_separate      , step_size, leng, msg + "(Separate) Cartesian to $\\phi$", sampling, gen_data_only, plot_only)
    #if mode ==  8 or plot_only: benchmark(create_cartesian_separate, fx.ptetaphi_separate , step_size, leng, msg + "(Separate) Cartesian to $p_T, \\eta, \\phi $", sampling, gen_data_only, plot_only)
    #if mode ==  9 or plot_only: benchmark(create_cartesian_separate, fx.ptetaphie_separate, step_size, leng, msg + "(Separate) Cartesian to $p_T, \\eta, \\phi E$", sampling, gen_data_only, plot_only)
    #if mode == 10 or plot_only: benchmark(create_polar_combined, fx.px_combined     , step_size, leng, msg + "(Combined) Polar to Px", sampling, gen_data_only, plot_only)
    #if mode == 11 or plot_only: benchmark(create_polar_combined, fx.py_combined     , step_size, leng, msg + "(Combined) Polar to Py", sampling, gen_data_only, plot_only)
    #if mode == 12 or plot_only: benchmark(create_polar_combined, fx.pz_combined     , step_size, leng, msg + "(Combined) Polar to Pz", sampling, gen_data_only, plot_only)
    #if mode == 13 or plot_only: benchmark(create_polar_combined, fx.pxpypz_combined , step_size, leng, msg + "(Combined) Polar to PxPyPz", sampling, gen_data_only, plot_only)
    #if mode == 14 or plot_only: benchmark(create_polar_combined, fx.pxpypze_combined, step_size, leng, msg + "(Combined) Polar to PxPyPzE", sampling, gen_data_only, plot_only)
    #if mode == 15 or plot_only: benchmark(create_cartesian_combined, fx.pt_combined       , step_size, leng, msg + "(Combined) Cartesian to $p_T$", sampling, gen_data_only, plot_only)
    #if mode == 16 or plot_only: benchmark(create_cartesian_combined, fx.eta_combined      , step_size, leng, msg + "(Combined) Cartesian to $\\eta$", sampling, gen_data_only, plot_only)
    #if mode == 17 or plot_only: benchmark(create_cartesian_combined, fx.phi_combined      , step_size, leng, msg + "(Combined) Cartesian to $\\phi$", sampling, gen_data_only, plot_only)
    #if mode == 18 or plot_only: benchmark(create_cartesian_combined, fx.ptetaphi_combined , step_size, leng, msg + "(Combined) Cartesian to $p_T, \\eta, \\phi$", sampling, gen_data_only, plot_only)
    #if mode == 19 or plot_only: benchmark(create_cartesian_combined, fx.ptetaphie_combined, step_size, leng, msg + "(Combined) Cartesian to $p_T, \\eta, \\phi, E$", sampling, gen_data_only, plot_only)

    # --------------- Physics --------------------------- #
    if mode == 20 or plot_only: benchmark(create_cartesian_separate, fx.    p2_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to $P^{2}$", sampling, gen_data_only, plot_only)
    if mode == 21 or plot_only: benchmark(create_cartesian_separate, fx.     p_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to P", sampling, gen_data_only, plot_only)
    if mode == 22 or plot_only: benchmark(create_cartesian_separate, fx. beta2_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to $\\beta^{2}$", sampling, gen_data_only, plot_only)
    if mode == 23 or plot_only: benchmark(create_cartesian_separate, fx.  beta_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to $\\beta$", sampling, gen_data_only, plot_only)
    if mode == 24 or plot_only: benchmark(create_cartesian_separate, fx.    m2_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to $M^{2}$", sampling, gen_data_only, plot_only)
    if mode == 25 or plot_only: benchmark(create_cartesian_separate, fx.     m_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to M", sampling, gen_data_only, plot_only)
    if mode == 26 or plot_only: benchmark(create_cartesian_separate, fx.   mt2_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to $M_{T}^{2}$", sampling, gen_data_only, plot_only)
    if mode == 27 or plot_only: benchmark(create_cartesian_separate, fx.    mt_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to $M_{T}$", sampling, gen_data_only, plot_only)
    if mode == 28 or plot_only: benchmark(create_cartesian_separate, fx. theta_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to $\\theta$", sampling, gen_data_only, plot_only)
    if mode == 29 or plot_only: benchmark(create_cartesian_separate, fx.deltaR_cartesian_separate, step_size, leng, msg + "(Separate) Cartesian to $\\Delta R$", sampling, gen_data_only, plot_only)

    if mode == 30 or plot_only: benchmark(create_polar_separate, fx.    p2_polar_separate, step_size, leng, msg + "(Separate) Polar to $P^{2}$", sampling, gen_data_only, plot_only)
    if mode == 31 or plot_only: benchmark(create_polar_separate, fx.     p_polar_separate, step_size, leng, msg + "(Separate) Polar to P", sampling, gen_data_only, plot_only)
    if mode == 32 or plot_only: benchmark(create_polar_separate, fx. beta2_polar_separate, step_size, leng, msg + "(Separate) Polar to $\\beta^{2}$", sampling, gen_data_only, plot_only)
    if mode == 33 or plot_only: benchmark(create_polar_separate, fx.  beta_polar_separate, step_size, leng, msg + "(Separate) Polar to $\\beta$", sampling, gen_data_only, plot_only)
    if mode == 34 or plot_only: benchmark(create_polar_separate, fx.    m2_polar_separate, step_size, leng, msg + "(Separate) Polar to $M^{2}$", sampling, gen_data_only, plot_only)
    if mode == 35 or plot_only: benchmark(create_polar_separate, fx.     m_polar_separate, step_size, leng, msg + "(Separate) Polar to M", sampling, gen_data_only, plot_only)
    if mode == 36 or plot_only: benchmark(create_polar_separate, fx.   mt2_polar_separate, step_size, leng, msg + "(Separate) Polar to $M_{T}^{2}$", sampling, gen_data_only, plot_only)
    if mode == 37 or plot_only: benchmark(create_polar_separate, fx.    mt_polar_separate, step_size, leng, msg + "(Separate) Polar to $M_{T}$", sampling, gen_data_only, plot_only)
    if mode == 38 or plot_only: benchmark(create_polar_separate, fx. theta_polar_separate, step_size, leng, msg + "(Separate) Polar to $\\theta$", sampling, gen_data_only, plot_only)
    if mode == 39 or plot_only: benchmark(create_polar_separate, fx.deltaR_polar_separate, step_size, leng, msg + "(Separate) Polar to $\\Delta R$", sampling, gen_data_only, plot_only)

    if mode == 40 or plot_only: benchmark(create_cartesian_combined, fx.    p2_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to $P^{2}$", sampling, gen_data_only, plot_only)
    if mode == 41 or plot_only: benchmark(create_cartesian_combined, fx.     p_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to P", sampling, gen_data_only, plot_only)
    if mode == 42 or plot_only: benchmark(create_cartesian_combined, fx. beta2_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to $\\beta^{2}$", sampling, gen_data_only, plot_only)
    if mode == 43 or plot_only: benchmark(create_cartesian_combined, fx.  beta_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to $\\beta$", sampling, gen_data_only, plot_only)
    if mode == 44 or plot_only: benchmark(create_cartesian_combined, fx.    m2_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to $M^{2}$", sampling, gen_data_only, plot_only)
    if mode == 45 or plot_only: benchmark(create_cartesian_combined, fx.     m_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to M", sampling, gen_data_only, plot_only)
    if mode == 46 or plot_only: benchmark(create_cartesian_combined, fx.   mt2_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to $M_{T}^{2}$", sampling, gen_data_only, plot_only)
    if mode == 47 or plot_only: benchmark(create_cartesian_combined, fx.    mt_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to $M_{T}$", sampling, gen_data_only, plot_only)
    if mode == 48 or plot_only: benchmark(create_cartesian_combined, fx. theta_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to $\\theta$", sampling, gen_data_only, plot_only)
    if mode == 49 or plot_only: benchmark(create_cartesian_combined, fx.deltaR_cartesian_combined, step_size, leng, msg + "(Combined) Cartesian to $\\Delta R$", sampling, gen_data_only, plot_only)

    if mode == 50 or plot_only: benchmark(create_polar_combined, fx.    p2_polar_combined, step_size, leng, msg + "(Combined) Polar to $P^{2}$", sampling, gen_data_only, plot_only)
    if mode == 51 or plot_only: benchmark(create_polar_combined, fx.     p_polar_combined, step_size, leng, msg + "(Combined) Polar to P", sampling, gen_data_only, plot_only)
    if mode == 52 or plot_only: benchmark(create_polar_combined, fx. beta2_polar_combined, step_size, leng, msg + "(Combined) Polar to $\\beta^{2}$", sampling, gen_data_only, plot_only)
    if mode == 53 or plot_only: benchmark(create_polar_combined, fx.  beta_polar_combined, step_size, leng, msg + "(Combined) Polar to $\\beta$", sampling, gen_data_only, plot_only)
    if mode == 54 or plot_only: benchmark(create_polar_combined, fx.    m2_polar_combined, step_size, leng, msg + "(Combined) Polar to $M^{2}$", sampling, gen_data_only, plot_only)
    if mode == 55 or plot_only: benchmark(create_polar_combined, fx.     m_polar_combined, step_size, leng, msg + "(Combined) Polar to M", sampling, gen_data_only, plot_only)
    if mode == 56 or plot_only: benchmark(create_polar_combined, fx.   mt2_polar_combined, step_size, leng, msg + "(Combined) Polar to $M_{T}^{2}$", sampling, gen_data_only, plot_only)
    if mode == 57 or plot_only: benchmark(create_polar_combined, fx.    mt_polar_combined, step_size, leng, msg + "(Combined) Polar to $M_{T}$", sampling, gen_data_only, plot_only)
    if mode == 58 or plot_only: benchmark(create_polar_combined, fx. theta_polar_combined, step_size, leng, msg + "(Combined) Polar to $\\theta$", sampling, gen_data_only, plot_only)
    if mode == 59 or plot_only: benchmark(create_polar_combined, fx.deltaR_polar_combined, step_size, leng, msg + "(Combined) Polar to $\\Delta R$", sampling, gen_data_only, plot_only)

    if mode == 60 or plot_only: benchmark(create_matrix_4x4   , fx.dot_operator        , 1000, leng, msg + "$A \\cdot B$", sampling, gen_data_only, plot_only)
    if mode == 61 or plot_only: benchmark(create_symmetric_3x3, fx.eigenvalue_operator , 1000, leng, msg + "$Ax = \\lamba x$", sampling, gen_data_only, plot_only)
    if mode == 62 or plot_only: benchmark(create_matrix_3x3   , fx.determinant_operator, 1000, leng, msg + "det(A)", sampling, gen_data_only, plot_only)
    if mode == 63 or plot_only: benchmark(create_matrix_3x3   , fx.cofactor_operator   , 1000, leng, msg + "cofactor", sampling, gen_data_only, plot_only)
    if mode == 64 or plot_only: benchmark(create_matrix_3x3   , fx.inverse_operator    , 1000, leng, msg + "$A^{-1}$", sampling, gen_data_only, plot_only)
    if mode == 65 or plot_only: benchmark(create_cartesian_combined, fx.basematrix_nusol, step_size, leng, msg + "H-Matrix", sampling, gen_data_only, plot_only)

if __name__ == "__main__":
    import pyc
    import sys

    pyc = pyc.pyc("../build/pyc/interface")
    dev_name = "./v100"
    start(-1)
