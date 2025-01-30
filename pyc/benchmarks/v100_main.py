import pyc
import sys
sys.path.append("/data/gpfs/projects/punim0011/tnom6927/analysisg/anag-v100/pyc/build/pyc/")

import main
main.pyc = pyc.pyc("/data/gpfs/projects/punim0011/tnom6927/analysisg/anag-v100/pyc/build/pyc/interface")
main.dev_name = "./v100"
main.start(sys.argv[1])
