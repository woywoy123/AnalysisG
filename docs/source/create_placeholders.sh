#!/bin/bash

# Create placeholders in api/core
cat << 'INNER_EOF' > api/core/minimal_working_example.rst
Minimal Working Example
=======================

*This section is a placeholder for a minimal working example.*
INNER_EOF

# Create placeholders in api/modules
cat << 'INNER_EOF' > api/modules/mergecast.rst
mergecast
=========

*This section is a placeholder for mergecast documentation.*
INNER_EOF

# Create placeholders in api
cat << 'INNER_EOF' > api/interfaces_cython.rst
Cython Interfaces
=================

*This section is a placeholder for linking C++ and Cython code.*
INNER_EOF

cat << 'INNER_EOF' > api/cmake_cybuild.rst
CMake Tooling & cybuild
=======================

*This section is a placeholder for how to link C++ and Cython using cybuild.*
INNER_EOF

cat << 'INNER_EOF' > api/exports_hdf5.rst
Exporting graph_t to HDF5
=========================

*This section is a placeholder for HDF5 exports.*
INNER_EOF

cat << 'INNER_EOF' > api/exports_root.rst
Exporting to ROOT
=================

*This section is a placeholder for ROOT exports via writer.*
INNER_EOF

cat << 'INNER_EOF' > api/custom_root_pcm.rst
ROOT to PCM Mapping
===================

*This section is a placeholder for custom ROOT -> PCM maps.*
INNER_EOF

cat << 'INNER_EOF' > api/custom_root_types.rst
Adding Data Types to ROOT IO
============================

*This section is a placeholder for adding new data types to ROOT IO.*
INNER_EOF

cat << 'INNER_EOF' > api/custom_tensor_types.rst
Adding Tensor Types
===================

*This section is a placeholder for adding new tensor types (see graph_template).*
INNER_EOF

cat << 'INNER_EOF' > api/custom_tensor_casting.rst
Tensor Casting
==============

*This section is a placeholder for tensor casting.*
INNER_EOF

# Create placeholders in api/pyc
cat << 'INNER_EOF' > api/pyc/missing_cuda.rst
Missing CPU and CUDA Versions
=============================

*This section is a placeholder for handling missing CPU/CUDA versions.*
INNER_EOF

chmod +x create_placeholders.sh
./create_placeholders.sh
