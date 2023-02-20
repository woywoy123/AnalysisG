# PyTorchCustom
## Introduction:
This part of the AnalysisTopGNN repository is used to write PyTorch extension code in a modular way. 
The primary focus is to port and unify already written code in the repository into one single package, such that alternative algorithms can easily integrate highly optimized Physics centric CUDA/C++ code.

## Packages:
- Transform: A namespace used to transform between ATLAS's coordinate system back to cartesian and vice versa.
- Operators: Optimized PyTorch code used to apply to arbitrary vectors, and speed up common function calls, such as dot products. 
- Physics: A namesapce used to write Physics centric functions, that are used to extract relevant quantities from vectors.
- NuRecon: A namespace dedicated to Single and Double Neutrino reconstruction. The algorithm is a reimplementation of arXiv: 1305.1878v2.

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
- Description: Conversion to Cartesian System
```python 
Px(pt, phi)
```
```python 
Py(pt, phi)
```
```python 
Pz(pt, eta)
```
-- Output: l x 1 Tensor 
- Description: Conversion to Cartesian System simultaneously 
```python 
PxPyPz(pt, eta, phi)
```
-- Output: l x 3 Tensor 

- Description: Conversion to ATLAS System
```python 
PT(px, py)
```
```python 
Eta(px, py, pz)
```
```python 
Phi(px, py)
```
-- Output: l x 1 Tensor 
- Description: Conversion to ATLAS System simultaneously 
```python 
PtEtaPhi(px, py, pz)
```
-- Output: l x 3 Tensor 

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
// ------- Transform Cartesian Methods ------- //
const torch::Tensor TransformCUDA::Px(torch::Tensor pt, torch::Tensor phi); 
const torch::Tensor TransformCUDA::Py(torch::Tensor pt, torch::Tensor phi); 
const torch::Tensor TransformCUDA::Pz(torch::Tensor pt, torch::Tensor eta); 
const torch::Tensor TransformCUDA::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);

// CUDA Kernel Callers 
torch::Tensor _Px(torch::Tensor _pt, torch::Tensor _phi); 
torch::Tensor _Py(torch::Tensor _pt, torch::Tensor _phi); 
torch::Tensor _Pz(torch::Tensor _pt, torch::Tensor _eta); 
torch::Tensor _PxPyPz(torch::Tensor _pt, torch::Tensor _eta, torch::Tensor _phi); 

// ------- Transform Polar Methods ------- //
const torch::Tensor TransformCUDA::PT(torch::Tensor px, torch::Tensor py); 
const torch::Tensor TransformCUDA::Phi(torch::Tensor px, torch::Tensor py); 
const torch::Tensor TransformCUDA::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
const torch::Tensor TransformCUDA::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz);

// CUDA Kernel Callers 
torch::Tensor _Pt(torch::Tensor _px, torch::Tensor _py); 
torch::Tensor _Phi(torch::Tensor _px, torch::Tensor _py); 
torch::Tensor _Eta(torch::Tensor _px, torch::Tensor _py, torch::Tensor _pz); 
torch::Tensor _PtEtaPhi(torch::Tensor _px, torch::Tensor _py, torch::Tensor _pz); 

// ------- Operator Methods ------- //
const torch::Tensor OperatorsCUDA::CosTheta(torch::Tensor v1, torch::Tensor v2); 
const torch::Tensor OperatorsCUDA::SinTheta(torch::Tensor v1, torch::Tensor v2); 
const torch::Tensor OperatorsCUDA::_SinTheta(torch::Tensor cos);
const torch::Tensor OperatorsCUDA::Rx(torch::Tensor angle); 
const torch::Tensor OperatorsCUDA::Ry(torch::Tensor angle); 
const torch::Tensor OperatorsCUDA::Rz(torch::Tensor angle); 
const torch::Tensor OperatorsCUDA::Dot(torch::Tensor v1, torch::Tensor v2); 
const torch::Tensor OperatorsCUDA::Mul(torch::Tensor v1, torch::Tensor v2); 
const torch::Tensor OperatorsCUDA::Cofactors(torch::Tensor v1); 
const torch::Tensor OperatorsCUDA::Determinant(torch::Tensor Cofactors, torch::Tensor Matrix); 
const torch::Tensor OperatorsCUDA::Inverse(torch::Tensor Cofactors, torch::Tensor dets); 
const torch::Tensor OperatorsCUDA::Inv(torch::Tensor Matrix);
const torch::Tensor OperatorsCUDA::Det(torch::Tensor Matrix);  

// CUDA Kernel Callers
torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _Rx(torch::Tensor angle); 
torch::Tensor _Ry(torch::Tensor angle); 
torch::Tensor _Rz(torch::Tensor angle);
torch::Tensor _Mul(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _Cofactors(torch::Tensor v1); 
torch::Tensor _Determinant(torch::Tensor cofact, torch::Tensor Matrix); 
torch::Tensor _Inverse(torch::Tensor cofact, torch::Tensor Dets); 
torch::Tensor _inv(torch::Tensor Matrix); 
torch::Tensor _det(torch::Tensor Matrix); 

// ------- Neutrino Reconstruction Methods ------- //
// --- Neutrino Solution Methods --- //
const torch::Tensor NuSolCUDA::Solutions(std::vector<torch::Tensor> b_p, std::vector<torch::Tensor> b_C, std::vector<torch::Tensor> mu_P, std::vector<torch::Tensor> mu_C, torch::Tensor massT2, torch::Tensor massW2, torch::Tensor massNu2);
const torch::Tensor NuSolCUDA::V0(torch::Tensor metx, torch::Tensor mety); 
const torch::Tensor NuSolCUDA::H_Matrix(torch::Tensor Sols_, std::vector<torch::Tensor> b_C, std::vector<torch::Tensor> mu_C, torch::Tensor mu_P, torch::Tensor mu_phi); 
const torch::Tensor NuSolCUDA::Derivative(torch::Tensor x); 
const std::vector<torch::Tensor> NuSolCUDA::Intersection(torch::Tensor A, torch::Tensor B, double cutoff);

// --- Single Neutrino Algorithm Methods --- //
const torch::Tensor SingleNuCUDA::Sigma2(torch::Tensor Sxx, torch::Tensor Sxy, torch::Tensor Syx, torch::Tensor Syy); 
const std::vector<torch::Tensor> SingleNuCUDA::Nu(torch::Tensor b, torch::Tensor mu, torch::Tensor met, torch::Tensor phi, torch::Tensor Sxx, torch::Tensor Sxy, torch::Tensor Syx, torch::Tensor Syy, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu, double cutoff); 

// --- Double Neutrino Algorithm Methods --- //
const torch::Tensor DoubleNuCUDA::H_perp(torch::Tensor H);
const torch::Tensor DoubleNuCUDA::N(torch::Tensor H); 
const std::vector<torch::Tensor> DoubleNuCUDA::NuNu(torch::Tensor b, torch::Tensor b_, torch::Tensor mu, torch::Tensor mu_, torch::Tensor met, torch::Tensor phi, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu, double cutoff);

// CUDA Kernel Callers 
torch::Tensor _Solutions(torch::Tensor _muP2, torch::Tensor _bP2, torch::Tensor _mu_e, torch::Tensor _b_e, torch::Tensor _cos, torch::Tensor _sin, torch::Tensor mT2, torch::Tensor mW2, torch::Tensor mNu2); 
torch::Tensor _H_Matrix(torch::Tensor x1, torch::Tensor y1, torch::Tensor Z, torch::Tensor Om, torch::Tensor w, std::vector<torch::Tensor> b_C, torch::Tensor mu_phi, torch::Tensor mu_pz, torch::Tensor mu_P, torch::Tensor mu_theta, torch::Tensor Rx, torch::Tensor Ry, torch::Tensor Rz); 
torch::Tensor _H_Matrix(torch::Tensor sols, torch::Tensor mu_P); 
torch::Tensor _Pi_2(torch::Tensor v2); 
torch::Tensor _Unit(torch::Tensor v, std::vector<int> diag); 
torch::Tensor _Factorization(torch::Tensor G); 
torch::Tensor _Factorization(torch::Tensor G, torch::Tensor Q, torch::Tensor Cofactors); 
torch::Tensor _SwapXY(torch::Tensor G, torch::Tensor Q); 
std::vector<torch::Tensor> _EllipseLines(torch::Tensor Lines, torch::Tensor Q, torch::Tensor A, double cutoff); 
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
