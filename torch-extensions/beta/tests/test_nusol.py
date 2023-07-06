from nusol import NuSol
from common import *

def test_nusol_init():
    x = loadSingle()
    x = next(iter(x))
    ev, lep, bquark = x[0], x[1], x[2]
    nu = NuSol(bquark.vec, lep.vec, )
    print(lep.ten)


if __name__ == "__main__":
    test_nusol_init()

