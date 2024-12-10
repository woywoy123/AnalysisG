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

def create_matrix_4x4(dx, dx_old):
    if dx_old is None: return torch.randn((dx, 4, 4), dtype = torch.float64)
    data = torch.randn((dx, 4, 4), dtype = torch.float64)
    return torch.cat([data, dx_old], dim = 0)

def create_matrix_3x3(dx, dx_old):
    if dx_old is None: return torch.randn((dx, 3, 3), dtype = torch.float64)
    data = torch.randn((dx, 3, 3), dtype = torch.float64)
    return torch.cat([data, dx_old], dim = 0)

def create_symmetric_3x3(dx, dx_old):
    if dx_old is None: 
        tx = torch.randn((dx, 3, 3), dtype = torch.float64)
        tx += tx.transpose(1, 2).clone()
        return tx;

    data = torch.randn((dx, 3, 3), dtype = torch.float64)
    data += data.transpose(1, 2).clone()
    return torch.cat([data, dx_old], dim = 0)



if __name__ == "__main__":
    msg = "Performance Ratio Between Reference PyToch and Kernel Operator:"
    step_size = 10
    leng = 10

    # ------------------ Transformation --------------------------- #
    benchmark(create_polar_separate, px_separate     , step_size, leng, msg + " \n (Separate) Polar to Px")
    benchmark(create_polar_separate, py_separate     , step_size, leng, msg + " \n (Separate) Polar to Py")
    benchmark(create_polar_separate, pz_separate     , step_size, leng, msg + " \n (Separate) Polar to Pz")
    benchmark(create_polar_separate, pxpypz_separate , step_size, leng, msg + " \n (Separate) Polar to PxPyPz")
    benchmark(create_polar_separate, pxpypze_separate, step_size, leng, msg + " \n (Separate) Polar to PxPyPzE")

    benchmark(create_polar_combined, px_combined     , step_size, leng, msg + " \n (Combined) Polar to Px")
    benchmark(create_polar_combined, py_combined     , step_size, leng, msg + " \n (Combined) Polar to Py")
    benchmark(create_polar_combined, pz_combined     , step_size, leng, msg + " \n (Combined) Polar to Pz")
    benchmark(create_polar_combined, pxpypz_combined , step_size, leng, msg + " \n (Combined) Polar to PxPyPz")
    benchmark(create_polar_combined, pxpypze_combined, step_size, leng, msg + " \n (Combined) Polar to PxPyPzE")

    benchmark(create_cartesian_separate, pt_separate       , step_size, leng, msg + " \n (Separate) Cartesian to Pt")
    benchmark(create_cartesian_separate, eta_separate      , step_size, leng, msg + " \n (Separate) Cartesian to $\\eta$")
    benchmark(create_cartesian_separate, phi_separate      , step_size, leng, msg + " \n (Separate) Cartesian to $\\phi$")
    benchmark(create_cartesian_separate, ptetaphi_separate , step_size, leng, msg + " \n (Separate) Cartesian to Pt$\\eta\\phi$")
    benchmark(create_cartesian_separate, ptetaphie_separate, step_size, leng, msg + " \n (Separate) Cartesian to Pt$\\eta\\phi$E")

    benchmark(create_cartesian_combined, pt_combined       , step_size, leng, msg + " \n (Combined) Cartesian to Pt")
    benchmark(create_cartesian_combined, eta_combined      , step_size, leng, msg + " \n (Combined) Cartesian to $\\eta$")
    benchmark(create_cartesian_combined, phi_combined      , step_size, leng, msg + " \n (Combined) Cartesian to $\\phi$")
    benchmark(create_cartesian_combined, ptetaphi_combined , step_size, leng, msg + " \n (Combined) Cartesian to Pt$\\eta\\phi$")
    benchmark(create_cartesian_combined, ptetaphie_combined, step_size, leng, msg + " \n (Combined) Cartesian to Pt$\\eta\\phi$E")

    # ------------------ Physics --------------------------- #
    benchmark(create_cartesian_separate,     p2_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to $P^{2}$")
    benchmark(create_cartesian_separate,      p_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to P")
    benchmark(create_cartesian_separate,  beta2_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to $\\beta^{2}$")
    benchmark(create_cartesian_separate,   beta_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to $\\beta$")
    benchmark(create_cartesian_separate,     m2_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to $M^{2}$")
    benchmark(create_cartesian_separate,      m_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to M")
    benchmark(create_cartesian_separate,    mt2_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to $M_{T}^{2}$")
    benchmark(create_cartesian_separate,     mt_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to $M_{T}$")
    benchmark(create_cartesian_separate,  theta_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to $\\theta$")
    benchmark(create_cartesian_separate, deltaR_cartesian_separate, step_size, leng, msg + " \n (Separate) Cartesian to $\\Delta R$")

    benchmark(create_polar_separate,     p2_polar_separate, step_size, leng, msg + " \n (Separate) Polar to $P^{2}$")
    benchmark(create_polar_separate,      p_polar_separate, step_size, leng, msg + " \n (Separate) Polar to P")
    benchmark(create_polar_separate,  beta2_polar_separate, step_size, leng, msg + " \n (Separate) Polar to $\\beta^{2}$")
    benchmark(create_polar_separate,   beta_polar_separate, step_size, leng, msg + " \n (Separate) Polar to $\\beta$")
    benchmark(create_polar_separate,     m2_polar_separate, step_size, leng, msg + " \n (Separate) Polar to $M^{2}$")
    benchmark(create_polar_separate,      m_polar_separate, step_size, leng, msg + " \n (Separate) Polar to M")
    benchmark(create_polar_separate,    mt2_polar_separate, step_size, leng, msg + " \n (Separate) Polar to $M_{T}^{2}$")
    benchmark(create_polar_separate,     mt_polar_separate, step_size, leng, msg + " \n (Separate) Polar to $M_{T}$")
    benchmark(create_polar_separate,  theta_polar_separate, step_size, leng, msg + " \n (Separate) Polar to $\\theta$")
    benchmark(create_polar_separate, deltaR_polar_separate, step_size, leng, msg + " \n (Separate) Polar to $\\Delta R$")

    benchmark(create_cartesian_combined,     p2_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to $P^{2}$")
    benchmark(create_cartesian_combined,      p_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to P")
    benchmark(create_cartesian_combined,  beta2_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to $\\beta^{2}$")
    benchmark(create_cartesian_combined,   beta_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to $\\beta$")
    benchmark(create_cartesian_combined,     m2_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to $M^{2}$")
    benchmark(create_cartesian_combined,      m_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to M")
    benchmark(create_cartesian_combined,    mt2_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to $M_{T}^{2}$")
    benchmark(create_cartesian_combined,     mt_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to $M_{T}$")
    benchmark(create_cartesian_combined,  theta_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to $\\theta$")
    benchmark(create_cartesian_combined, deltaR_cartesian_combined, step_size, leng, msg + " \n (Combined) Cartesian to $\\Delta R$")

    benchmark(create_polar_combined,     p2_polar_combined, step_size, leng, msg + " \n (Combined) Polar to $P^{2}$")
    benchmark(create_polar_combined,      p_polar_combined, step_size, leng, msg + " \n (Combined) Polar to P")
    benchmark(create_polar_combined,  beta2_polar_combined, step_size, leng, msg + " \n (Combined) Polar to $\\beta^{2}$")
    benchmark(create_polar_combined,   beta_polar_combined, step_size, leng, msg + " \n (Combined) Polar to $\\beta$")
    benchmark(create_polar_combined,     m2_polar_combined, step_size, leng, msg + " \n (Combined) Polar to $M^{2}$")
    benchmark(create_polar_combined,      m_polar_combined, step_size, leng, msg + " \n (Combined) Polar to M")
    benchmark(create_polar_combined,    mt2_polar_combined, step_size, leng, msg + " \n (Combined) Polar to $M_{T}^{2}$")
    benchmark(create_polar_combined,     mt_polar_combined, step_size, leng, msg + " \n (Combined) Polar to $M_{T}$")
    benchmark(create_polar_combined,  theta_polar_combined, step_size, leng, msg + " \n (Combined) Polar to $\\theta$")
    benchmark(create_polar_combined, deltaR_polar_combined, step_size, leng, msg + " \n (Combined) Polar to $\\Delta R$")

    benchmark(create_matrix_4x4,  dot_operator, 1000, leng, msg + " \n $A \\cdot B$")
    benchmark(create_symmetric_3x3,  eigenvalue_operator, 1000, leng, msg + " \n $Ax = \\lamba x$")
    benchmark(create_matrix_3x3,  determinant_operator, 1000, leng, msg + " \n det(A)")
    benchmark(create_matrix_3x3,  cofactor_operator, 1000, leng, msg + " \n cofactor")
    benchmark(create_matrix_3x3,  inverse_operator, 1000, leng, msg + " \n $A^{-1}$")

    benchmark(create_cartesian_combined, basematrix_nusol, step_size, leng, msg + " \n H-Matrix")

