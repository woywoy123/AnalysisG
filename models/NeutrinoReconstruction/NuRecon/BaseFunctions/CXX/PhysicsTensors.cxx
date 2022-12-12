#include "../Headers/PhysicsTensors.h"

torch::Tensor __Slicer(torch::Tensor Vector, int sdim, int edim)
{
	return Vector.index({torch::indexing::Slice(), torch::indexing::Slice(sdim, edim, edim)});
}

torch::Tensor PhysicsTensors::ToPxPyPzE(torch::Tensor Vector)
{
	Vector = Vector.view({-1, 4}); 
	torch::Tensor pt = __Slicer(Vector, 0, 1);  
  	torch::Tensor eta =  __Slicer(Vector, 1, 2); 
  	torch::Tensor phi =  __Slicer(Vector, 2, 3); 
  	torch::Tensor e =  __Slicer(Vector, 3, 4); 

  	torch::Tensor px = pt*torch::cos(phi); 
  	torch::Tensor py = pt*torch::sin(phi); 
  	torch::Tensor pz = pt*torch::sinh(eta); 

  	return torch::cat({px, py, pz, e}, 1); 
}

torch::Tensor PhysicsTensors::Mass2Polar(torch::Tensor Polar )
{
	torch::Tensor Pmu = ToPxPyPzE(Polar); 
	return PhysicsTensors::Mass2Cartesian(Pmu); 
}

torch::Tensor PhysicsTensors::MassPolar(torch::Tensor Polar)
{
	return PhysicsTensors::Mass2Polar(Polar).sqrt(); 
}

torch::Tensor PhysicsTensors::Mass2Cartesian(torch::Tensor Cartesian)
{
	Cartesian = Cartesian.view({-1, 4}); 
	torch::Tensor px = __Slicer(Cartesian, 0, 1); 
	torch::Tensor py = __Slicer(Cartesian, 1, 2); 
	torch::Tensor pz = __Slicer(Cartesian, 2, 3); 
	torch::Tensor e  = __Slicer(Cartesian, 3, 4); 
	return (e.pow(2) - px.pow(2) - py.pow(2) - pz.pow(2)).abs();  
}

torch::Tensor PhysicsTensors::MassCartesian(torch::Tensor Cartesian)
{
	return PhysicsTensors::Mass2Cartesian(Cartesian).sqrt(); 
}

torch::Tensor PhysicsTensors::BetaCartesian( torch::Tensor Vector )
{
	torch::Tensor px = __Slicer(Vector, 0, 1);
	torch::Tensor py = __Slicer(Vector, 1, 2);
	torch::Tensor pz = __Slicer(Vector, 2, 3);
	torch::Tensor e  = __Slicer(Vector, 3, 4);
	return torch::sqrt((torch::pow(px, 2) + torch::pow(py, 2) + torch::pow(pz, 2)))/e; 
}

torch::Tensor PhysicsTensors::BetaPolar(torch::Tensor Vector)
{
	return PhysicsTensors::BetaCartesian(PhysicsTensors::ToPxPyPzE( Vector ));
}

torch::Tensor PhysicsTensors::CosThetaCartesian( torch::Tensor v1, torch::Tensor v2 )
{
	torch::Tensor v1_2 = torch::sqrt(torch::pow(v1, 2).sum({-1}, true)); 
	torch::Tensor v2_2 = torch::sqrt(torch::pow(v2, 2).sum({-1}, true)); 
	torch::Tensor dot = torch::sum(v1 * v2, {-1}, true); 
	return dot/(v1_2 * v2_2); 
}





