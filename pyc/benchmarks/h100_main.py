import pyc
import sys
sys.path.append("/data/gpfs/projects/punim0011/tnom6927/analysisg/anag-h100/pyc/build/pyc/")

import main
main.pyc = pyc.pyc("/data/gpfs/projects/punim0011/tnom6927/analysisg/anag-h100/pyc/build/pyc/interface")
main.dev_name = "./h100"
main.start(sys.argv[1])
