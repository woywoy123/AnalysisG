import time
import pyc
import torch
pyx = pyc.pyc()

tolerance = 10e-7

def test_this(fx, data):
    t1 = time.time()
    tx = fx(*data)
    return time.time() - t1, tx

def compare(txc, txp):
    msk_t = txp == 0
    msk_c = txc == 0
    try: assert not (msk_t != msk_c).reshape(-1).sum(-1)
    except: return
    msk_t *= msk_c
    dif = abs(txc - txp)[msk_t == False]/txp[msk_t == False]
    assert not (dif > tolerance).view(-1).sum(-1)

def px_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_px, [cpu_data[0] , cpu_data[2] ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_px, [cuda_data[0], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_px  , [cuda_data[0], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def py_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_py, [cpu_data[0] , cpu_data[2] ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_py, [cuda_data[0], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_py  , [cuda_data[0], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def pz_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_pz, [cpu_data[0] , cpu_data[1] ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_pz, [cuda_data[0], cuda_data[1]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_pz  , [cuda_data[0], cuda_data[1]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def pxpypz_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_pxpypz, [cpu_data[0] , cpu_data[1] , cpu_data[2]])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_pxpypz, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_pxpypz  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def pxpypze_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_pxpypze, [ cpu_data[0],  cpu_data[1],  cpu_data[2],  cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_pxpypze, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_pxpypze  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def px_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_px, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_px, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_px  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def py_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_py, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_py, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_py  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def pz_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_pz, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_pz, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_pz  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def pxpypz_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_pxpypz, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_pxpypz, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_pxpypz  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def pxpypze_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_pxpypze, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_pxpypze, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_pxpypze  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def pt_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_pt, [cpu_data[0] , cpu_data[1] ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_pt, [cuda_data[0], cuda_data[1]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_pt  , [cuda_data[0], cuda_data[1]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def eta_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_eta, [cpu_data[0] , cpu_data[1] , cpu_data[2] ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_eta, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_eta  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def phi_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_pz, [cpu_data[0] , cpu_data[1] ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_pz, [cuda_data[0], cuda_data[1]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_pz  , [cuda_data[0], cuda_data[1]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def ptetaphi_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_ptetaphi, [cpu_data[0] , cpu_data[1] , cpu_data[2]])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_ptetaphi, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_ptetaphi  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def ptetaphie_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_separate_ptetaphie, [ cpu_data[0],  cpu_data[1],  cpu_data[2],  cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_transform_separate_ptetaphie, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_transform_separate_ptetaphie  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def pt_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_pt, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_pt, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_pt  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def eta_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_eta, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_eta, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_eta  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def phi_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_phi, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_phi, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_phi  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def ptetaphi_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_ptetaphi, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_ptetaphi, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_ptetaphi   , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def ptetaphie_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_transform_combined_ptetaphie, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_transform_combined_ptetaphie, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_transform_combined_ptetaphie  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)






def p2_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_p2, [cpu_data[0] , cpu_data[1] , cpu_data[2] ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_p2, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_p2  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def p_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_p, [cpu_data[0] , cpu_data[1] , cpu_data[2] ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_p, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_p  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def beta2_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_beta2, [cpu_data[0] , cpu_data[1] , cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_beta2, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_beta2  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def beta_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_beta, [cpu_data[0] , cpu_data[1] , cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_beta, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_beta  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def m2_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_m2, [cpu_data[0] , cpu_data[1] , cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_m2, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_m2  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def m_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_m, [cpu_data[0] , cpu_data[1] , cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_m, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_m  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def mt2_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_mt2, [cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_mt2, [cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_mt2  , [cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def mt_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_mt, [cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_mt, [cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_mt  , [cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def theta_cartesian_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_theta, [cpu_data[0], cpu_data[1] , cpu_data[2]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_theta, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_theta  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def deltaR_cartesian_separate(cpu_data):
    randx = torch.randperm(cpu_data[0].size(0))
    randx = [i[randx] for i in cpu_data]
    cuda_data = [i.to(device = "cuda") for i in cpu_data]
    cuda_rand = [i.to(device = "cuda") for i in randx]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_separate_deltaR, sum([[cpu_data[i] , randx[i]] for i in range(3)], []))
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_separate_deltaR, sum([[cuda_data[i], cuda_rand[i]] for i in range(3)], []))
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_separate_deltaR  , sum([[cuda_data[i], cuda_rand[i]] for i in range(3)], []))
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def p2_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_p2, [cpu_data[0] , cpu_data[1] , cpu_data[2] ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_p2, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_p2  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def p_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_p, [cpu_data[0] , cpu_data[1] , cpu_data[2] ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_p, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_p  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def beta2_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_beta2, [cpu_data[0] , cpu_data[1] , cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_beta2, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_beta2  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def beta_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_beta, [cpu_data[0] , cpu_data[1] , cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_beta, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_beta  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def m2_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_m2, [cpu_data[0] , cpu_data[1] , cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_m2, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_m2  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def m_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_m, [cpu_data[0] , cpu_data[1] , cpu_data[2] , cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_m, [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_m  , [cuda_data[0], cuda_data[1], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def mt2_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_mt2, [cpu_data[0] , cpu_data[2] ,  cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_mt2, [cuda_data[0], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_mt2  , [cuda_data[0], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def mt_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_mt, [cpu_data[0] , cpu_data[2] ,  cpu_data[3]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_mt, [cuda_data[0], cuda_data[2], cuda_data[3]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_mt  , [cuda_data[0], cuda_data[2], cuda_data[3]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def theta_polar_separate(cpu_data):
    cuda_data = [i.to(device = "cuda") for i in cpu_data]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_theta, [cpu_data[0], cpu_data[1] , cpu_data[2]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_theta, [cuda_data[0], cuda_data[1], cuda_data[2]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_theta  , [cuda_data[0], cuda_data[1], cuda_data[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def deltaR_polar_separate(cpu_data):
    randx = torch.randperm(cpu_data[0].size(0))
    randx = [i[randx] for i in cpu_data]
    cuda_data = [i.to(device = "cuda") for i in cpu_data]
    cuda_rand = [i.to(device = "cuda") for i in randx]

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_separate_deltaR, [cpu_data[1] ,     randx[1], cpu_data[2] ,     randx[2]])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_separate_deltaR, [cuda_data[1], cuda_rand[1], cuda_data[2], cuda_rand[2]])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_separate_deltaR  , [cuda_data[1], cuda_rand[1], cuda_data[2], cuda_rand[2]])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def p2_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_p2, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_p2, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_p2  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def p_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_p, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_p, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_p  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def beta2_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_beta2, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_beta2, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_beta2  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def beta_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_beta, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_beta, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_beta  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def m2_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_m2, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_m2, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_m2  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def m_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_m, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_m, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_m  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def mt2_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_mt2, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_mt2, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_mt2  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def mt_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_mt, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_mt, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_mt  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def theta_cartesian_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_theta, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_theta, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_theta  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def deltaR_cartesian_combined(cpu_data):
    randx = torch.randperm(cpu_data.size(0))
    randx = cpu_data[randx]
    cuda_data = cpu_data.to(device = "cuda")
    cuda_rand = randx.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_cartesian_combined_deltaR, [cpu_data,      randx])
    tcuda_dt, txc = test_this(pyx.tensor_physics_cartesian_combined_deltaR, [cuda_data, cuda_rand])
    cuda_dt , txp = test_this(pyx.cuda_physics_cartesian_combined_deltaR  , [cuda_data, cuda_rand])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)


def p2_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_p2, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_p2, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_p2  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def p_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_p, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_p, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_p  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def beta2_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_beta2, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_beta2, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_beta2  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def beta_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_beta, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_beta, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_beta  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def m2_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_m2, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_m2, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_m2  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def m_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_m, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_m, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_m  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def mt2_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_mt2, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_mt2, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_mt2  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def mt_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_mt, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_mt, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_mt  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def theta_polar_combined(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_theta, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_theta, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_theta  , [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def deltaR_polar_combined(cpu_data):
    randx = cpu_data[torch.randperm(cpu_data.size(0))]
    cuda_data = cpu_data.to(device = "cuda")
    cuda_rand = randx.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_physics_polar_combined_deltaR, [cpu_data , randx    ])
    tcuda_dt, txc = test_this(pyx.tensor_physics_polar_combined_deltaR, [cuda_data, cuda_rand])
    cuda_dt , txp = test_this(pyx.cuda_physics_polar_combined_deltaR  , [cuda_data, cuda_rand])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def dot_operator(cpu_data):
    randx = cpu_data[torch.randperm(cpu_data.size(0))]
    cuda_data = cpu_data.to(device = "cuda")
    cuda_rand = randx.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_operators_dot, [cpu_data , randx    ])
    tcuda_dt, txc = test_this(pyx.tensor_operators_dot, [cuda_data, cuda_rand])
    cuda_dt , txp = test_this(  pyx.cuda_operators_dot, [cuda_data, cuda_rand])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def eigenvalue_operator(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(torch.linalg.eig, [cpu_data ])
    tcuda_dt, txc = test_this(torch.linalg.eig, [cuda_data])
    cuda_dt , txp = test_this(pyx.cuda_operators_eigenvalue, [cuda_data])

    crl = txc[0].real.sort(-1)[0]
    rl  = txp[0].sort(-1)[0]
    compare(crl, rl)

    crl = txc[0].imag.sort(-1)[0]
    rl  = txp[1].sort(-1)[0]
    compare(crl, rl)

    return (cpu_dt, tcuda_dt, cuda_dt)

def determinant_operator(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_operators_determinant, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_operators_determinant, [cuda_data])
    cuda_dt , txp = test_this(  pyx.cuda_operators_determinant, [cuda_data])
    compare(txc.view(-1), txp.view(-1))
    return (cpu_dt, tcuda_dt, cuda_dt)

def cofactor_operator(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_operators_cofactors, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_operators_cofactors, [cuda_data])
    cuda_dt , txp = test_this(  pyx.cuda_operators_cofactors, [cuda_data])
    compare(txc, txp)
    return (cpu_dt, tcuda_dt, cuda_dt)

def inverse_operator(cpu_data):
    cuda_data = cpu_data.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_operators_inverse, [cpu_data ])
    tcuda_dt, txc = test_this(pyx.tensor_operators_inverse, [cuda_data])
    cuda_dt , txp = test_this(  pyx.cuda_operators_inverse, [cuda_data])
    compare(txc, txp[0])
    return (cpu_dt, tcuda_dt, cuda_dt)

def basematrix_nusol(cpu_b):
    masses = torch.ones((cpu_b.size(0), 3), dtype = torch.float64)
    masses[:, 0] *= 172*1000
    masses[:, 1] *= 82*1000
    masses[:, 2] *= 0

    cpu_l = cpu_b[torch.randperm(cpu_b.size(0))]

    cuda_b = cpu_b.to(device = "cuda")
    cuda_l = cpu_l.to(device = "cuda")
    cuda_m = masses.to(device = "cuda")

    cpu_dt  ,   _ = test_this(pyx.tensor_nusol_base_basematrix, [cpu_b,  cpu_l, masses])
    tcuda_dt, txc = test_this(pyx.tensor_nusol_base_basematrix, [cuda_b, cuda_l, cuda_m])
    cuda_dt , txp = test_this(  pyx.cuda_nusol_base_basematrix, [cuda_b, cuda_l, cuda_m])

    ok = txp["passed"] == 1
    compare(txc[ok], txp["H_perp"][ok])
    return (cpu_dt, tcuda_dt, cuda_dt)
