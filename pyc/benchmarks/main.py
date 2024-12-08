from common import *
from fx import *
from core.plotting import TLine, TH1F

def benchmark(gen, fx, dx, lf, title, iters = 1000):
    df = {}
    data = None
    for i in range(lf):
        lx = dx + i*dx
        data  = gen(dx, data)
        df = merge(df, repeat(fx, data, iters, lx))

    tl = Line(title)
    tl.Lines = MakeLines(df)
    tl.xStep = dx
    tl.xMax = lx
    tl.xMin = dx
    tl.yMin = 0
    tl.Filename = fx.__name__
    tl.SaveFigure()

def create_polar_combined(dx, dx_old):
    if dx_old is None: return create_particle(dx)["polar"]
    data = create_particle(dx)["polar"]
    return torch.cat([data, dx_old], dim = 0)

def create_polar_separate(dx, dx_old):
    data = create_particle(dx)["polar"]
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

def create_cartesian_combined(dx, dx_old):
    if dx_old is None: return create_particle(dx)["cartesian"]
    data = create_particle(dx)["cartesian"]
    return torch.cat([data, dx_old], dim = 0)


def create_cartesian_separate(dx, dx_old):
    data = create_particle(dx)["cartesian"]
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

if __name__ == "__main__":
    msg = "Performance Ratio Between Reference PyToch and Kernel Operator:"
#    benchmark(create_polar_separate, px_separate     , 10, 10, msg + " \n (Separate) Polar to Px")
#    benchmark(create_polar_separate, py_separate     , 10, 10, msg + " \n (Separate) Polar to Py")
#    benchmark(create_polar_separate, pz_separate     , 10, 10, msg + " \n (Separate) Polar to Pz")
#    benchmark(create_polar_separate, pxpypz_separate , 10, 10, msg + " \n (Separate) Polar to PxPyPz")
#    benchmark(create_polar_separate, pxpypze_separate, 10, 10, msg + " \n (Separate) Polar to PxPyPzE")

#    benchmark(create_polar_combined, px_combined     , 10, 10, msg + " \n (Combined) Polar to Px")
#    benchmark(create_polar_combined, py_combined     , 10, 10, msg + " \n (Combined) Polar to Py")
#    benchmark(create_polar_combined, pz_combined     , 10, 10, msg + " \n (Combined) Polar to Pz")
#    benchmark(create_polar_combined, pxpypz_combined , 10, 10, msg + " \n (Combined) Polar to PxPyPz")
#    benchmark(create_polar_combined, pxpypze_combined, 10, 10, msg + " \n (Combined) Polar to PxPyPzE")

#    benchmark(create_cartesian_separate, pt_separate       , 10, 10, msg + " \n (Separate) Cartesian to Pt")
#    benchmark(create_cartesian_separate, eta_separate      , 10, 10, msg + " \n (Separate) Cartesian to $\\eta$")
#    benchmark(create_cartesian_separate, phi_separate      , 10, 10, msg + " \n (Separate) Cartesian to $\\phi$")
#    benchmark(create_cartesian_separate, ptetaphi_separate , 10, 10, msg + " \n (Separate) Cartesian to Pt$\\eta\\phi$")
#    benchmark(create_cartesian_separate, ptetaphie_separate, 10, 10, msg + " \n (Separate) Cartesian to Pt$\\eta\\phi$E")
#
#    benchmark(create_cartesian_combined, pt_combined       , 10, 10, msg + " \n (Combined) Cartesian to Pt")
#    benchmark(create_cartesian_combined, eta_combined      , 10, 10, msg + " \n (Combined) Cartesian to $\\eta$")
#    benchmark(create_cartesian_combined, phi_combined      , 10, 10, msg + " \n (Combined) Cartesian to $\\phi$")
    benchmark(create_cartesian_combined, ptetaphi_combined , 10, 10, msg + " \n (Combined) Cartesian to Pt$\\eta\\phi$")
    benchmark(create_cartesian_combined, ptetaphie_combined, 10, 10, msg + " \n (Combined) Cartesian to Pt$\\eta\\phi$E")


