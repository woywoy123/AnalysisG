# PyTorchCustom
## Introduction:
This part of the AnalysisTopGNN repository is used to write PyTorch extension code in a modular way. 
The primary focus is to port and unify already written code in the repository into one single package, such that alternative algorithms can easily integrate highly optimized Physics centric CUDA/C++ code.

## Packages:
- Transform: A namespace used to transform between ATLAS's coordinate system back to cartesian and vice versa.
- Operators: Optimized PyTorch code used to apply to arbitrary vectors, and speed up common function calls, such as dot products. 
- Physics: A namesapce used to write Physics centric functions, that are used to extract relevant quantities from vectors.

## How To Use:
```bash 
pip install . 
```

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

## C++ Namespace::Functions:
### CUDA Interfaces:
```C++
const torch::Tensor TransformCUDA::Px(torch::Tensor pt, torch::Tensor phi); 
const torch::Tensor TransformCUDA::Py(torch::Tensor pt, torch::Tensor phi); 
const torch::Tensor TransformCUDA::Pz(torch::Tensor pt, torch::Tensor eta); 
const torch::Tensor TransformCUDA::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

// CUDA Kernel Callers 
torch::Tensor _Px(torch::Tensor _pt, torch::Tensor _phi); 
torch::Tensor _Py(torch::Tensor _pt, torch::Tensor _phi); 
torch::Tensor _Pz(torch::Tensor _pt, torch::Tensor _eta); 
torch::Tensor _PxPyPz(torch::Tensor _pt, torch::Tensor _eta, torch::Tensor _phi); 

const torch::Tensor TransformCUDA::PT(torch::Tensor px, torch::Tensor py); 
const torch::Tensor TransformCUDA::Phi(torch::Tensor px, torch::Tensor py); 
const torch::Tensor TransformCUDA::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
const torch::Tensor TransformCUDA::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

// CUDA Kernel Callers 
torch::Tensor _Pt(torch::Tensor _px, torch::Tensor _py); 
torch::Tensor _Phi(torch::Tensor _px, torch::Tensor _py); 
torch::Tensor _Eta(torch::Tensor _px, torch::Tensor _py, torch::Tensor _pz); 
torch::Tensor _PtEtaPhi(torch::Tensor _px, torch::Tensor _py, torch::Tensor _pz); 

const torch::Tensor OperatorsCUDA::Dot(torch::Tensor v1, torch::Tensor v2); 
const torch::Tensor OperatorsCUDA::CosTheta(torch::Tensor v1, torch::Tensor v2); 
const torch::Tensor OperatorsCUDA::SinTheta(torch::Tensor v1, torch::Tensor v2); 
const torch::Tensor OperatorsCUDA::_SinTheta(torch::Tensor cos); 

// CUDA Kernel Callers
torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2); 
```

### Tensor Interfaces:
```C++ 
torch::Tensor TransformTensors::Px(torch::Tensor pt, torch::Tensor phi); 
torch::Tensor TransformTensors::Py(torch::Tensor pt, torch::Tensor phi); 
torch::Tensor TransformTensors::Pz(torch::Tensor pt, torch::Tensor eta); 
torch::Tensor TransformTensors::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 

torch::Tensor OperatorsTensors::Dot(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor OperatorsTensors::CosTheta(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor OperatorsTensors::SinTheta(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor OperatorsTensors::_SinTheta(torch::Tensor cos);

torch::Tensor TransformTensors::PT(torch::Tensor px, torch::Tensor py); 
torch::Tensor TransformTensors::Phi(torch::Tensor px, torch::Tensor py); 
torch::Tensor TransformTensors::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor TransformTensors::_Eta(torch::Tensor pt, torch::Tensor pz); 
std::vector<torch::Tensor> TransformFloats::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 

const torch::Tensor PhysicsCartesianTensors::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
const torch::Tensor PhysicsCartesianTensors::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
const torch::Tensor PhysicsCartesianTensors::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
const torch::Tensor PhysicsCartesianTensors::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
const torch::Tensor PhysicsCartesianTensors::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
const torch::Tensor PhysicsCartesianTensors::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
const torch::Tensor PhysicsCartesianTensors::Mt2(torch::Tensor pz, torch::Tensor e); 
const torch::Tensor PhysicsCartesianTensors::Mt(torch::Tensor pz, torch::Tensor e); 
const torch::Tensor PhysicsCartesianTensors::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
const torch::Tensor PhysicsCartesianTensors::DeltaR(torch::Tensor px1, torch::Tensor px2, torch::Tensor py1, torch::Tensor py2, torch::Tensor pz1, torch::Tensor pz2); 

const std::vector<torch::Tensor> _Transform(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
const torch::Tensor PhysicsPolarTensors::P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
const torch::Tensor PhysicsPolarTensors::P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
const torch::Tensor PhysicsPolarTensors::Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
const torch::Tensor PhysicsPolarTensors::Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
const torch::Tensor PhysicsPolarTensors::M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
const torch::Tensor PhysicsPolarTensors::M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
const torch::Tensor PhysicsPolarTensors::Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e); 
const torch::Tensor PhysicsPolarTensors::Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e); 
const torch::Tensor PhysicsPolarTensors::Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);
const torch::Tensor PhysicsPolarTensors::DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2); 

torch::Tensor PhysicsTensors::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor PhysicsTensors::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor PhysicsTensors::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
torch::Tensor PhysicsTensors::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
torch::Tensor PhysicsTensors::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
torch::Tensor PhysicsTensors::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
torch::Tensor PhysicsTensors::Mt2(torch::Tensor pz, torch::Tensor e); 
torch::Tensor PhysicsTensors::Mt(torch::Tensor pz, torch::Tensor e); 
torch::Tensor PhysicsTensors::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
torch::Tensor PhysicsTensors::DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2); 
```

### Float Interfaces:
```C++
double TransformFloats::Px(double pt, double phi); 
double TransformFloats::Py(double pt, double phi); 
double TransformFloats::Pz(double pt, double eta); 
std::vector<double> TransformFloats::PxPyPz(double pt, double eta, double phi); 

double TransformFloats::PT(double px, double py); 
double TransformFloats::Phi(double px, double py); 
double TransformFloats::Eta(double px, double py, double pz); 
std::vector<double> TransformFloats::PtEtaPhi(double px, double py, double pz); 
```
