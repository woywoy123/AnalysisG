# PyTorchCustom
## Introduction:
This part of the AnalysisTopGNN repository is used to write PyTorch extension code in a modular way. 
The primary focus is to port and unify already written code in the repository into one single package, such that alternative algorithms can easily integrate highly optimized Physics centric CUDA/C++ code.

## Packages:
- Transform: A namespace used to transform between ATLAS's coordinate system back to cartesian and vice versa.
- Operators: Optimized PyTorch code used to apply to arbitrary vectors, and speed up common function calls, such as dot products. 
- Physics: A namesapce used to write Physics centric functions, that are used to extract relevant quantities from vectors.

## How To Use:
``bash
pip install . 
``

## Python Importation Directory:
### Transform:
- PyC.Transform.Floats
- PyC.Transform.Tensors
- PyC.Transform.CUDA

#### Functions:
- Px: Convert to Px
- Py: Convert to Py
- Pz: Convert to Pz
- PxPyPz: Convert into Cartesian simultaneously
- PT: Convert to PT from Cartesian 
- Phi: Convert to Phi from Cartesian 
- Eta: Convert to Eta from Cartesian 
- PtEtaPhi: Convert into Polar simultaneously

### Operators:
- PyC.Operators.Tensors
- PyC.Operators.CUDA 

#### Functions:
- Dot(tensor, tensor): Dot product <<1, 2, 3>> * <<1, 2, 3>> = <<1 + 4 + 9>>
- CosTheta(tensor, tensor): Cos(theta) of two vectors 
- SinTheta(tensor, tensor): sqrt(1 - Cos(theta)^2)

### Physics:
- PyC.Physics.Tensors.Cartesian
- PyC.Physics.Tensors.Polar 

#### Functions:
- P2: 
- P:
- Beta2:
- Beta:
- M2: 
- M: 
- Mt2: 
- Theta: 
- DeltaR: 

## C++ Namespaces:
``C++
OperatorsCUDA::Dot(torch::Tensor v1, torch::Tensor v2); 
``
