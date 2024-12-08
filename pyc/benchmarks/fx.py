import time
import pyc
pyx = pyc.pyc()

tolerance = 10e-7

def test_this(fx, data):
    t1 = time.time()
    tx = fx(*data)
    return time.time() - t1, tx

def compare(txc, txp): assert not (abs(txc - txp) > tolerance).view(-1).sum(-1)

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



















