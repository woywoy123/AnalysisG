import pyc
import sys

sys.path.append("../build/pyc/")

import main
main.pyc = pyc.pyc("../build/pyc/interface")
main.dev_name = "./gtx1080"
main.start(sys.argv[1])
