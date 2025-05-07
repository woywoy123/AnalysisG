/**
 * @file a100_main.py
 * @brief Entry point for running benchmarks on the A100 GPU.
 */

import pyc
import sys

# Add the path to the pyc module.
sys.path.append("/data/gpfs/projects/punim0011/tnom6927/analysisg/anag-a100/pyc/build/pyc/")

import main

# Initialize the pyc module with the specified interface path.
main.pyc = pyc.pyc("/data/gpfs/projects/punim0011/tnom6927/analysisg/anag-a100/pyc/build/pyc/interface")

# Set the device name for the benchmark.
main.dev_name = "./a100"

# Start the benchmark with the provided command-line arguments.
main.start(sys.argv[1])
