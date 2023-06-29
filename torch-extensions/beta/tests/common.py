import torch
import vector
import inspect

def create_vector_cartesian(px, py, pz, e):
    print("->", inspect.stack()[1][3])
    return vector.obj(px = px, py = py, pz = pz, E = e)

def create_vector_polar(pt, eta, phi, e):
    print("->", inspect.stack()[1][3])
    return vector.obj(pt = pt, eta = eta, phi = phi, E = e)

def create_tensor_cpu_1d(inpt = [1., 2., 3., 4.]):
    return torch.tensor(inpt).view(1, -1).to(dtype = torch.float64)

def create_tensor_cpu_nd(dim):
    return torch.cat([create_tensor_cpu_1d*(i+1) for i in range(dim)], 0)

def rounder(inpt1, inpt2, d = 5):
    try: i1 = round(inpt1.view(-1)[0].item(), d)
    except:
        try: i1 = round(inpt1[0], d)
        except: i1 = round(inpt1, d)
    i2 = round(inpt2, d)
    return i1 == i2

def rounder_l(inpt1, inpt2, d = 5):
    try: inpt1 = inpt1.view(-1).tolist()
    except:
        if isinstance(inpt1[0], list): inpt1 = inpt1[0]
        else: pass
    dif = sum([abs(round(i - j, d))/j for i, j in zip(inpt1, inpt2)])
    return dif*100 < 1


