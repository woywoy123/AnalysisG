#include <pyc/cupyc.h>
#include <graph/graph.cuh>
#include <nusol/nusol.cuh>
#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <transform/transform.cuh>
#include <operators/operators.cuh>

torch::Tensor pyc::transform::separate::Pt(torch::Tensor px, torch::Tensor py){
    changedev(&px); 
    return transform_::Pt(&px, &py); 
}

torch::Tensor pyc::transform::combined::Pt(torch::Tensor pmc){
    changedev(&pmc); 
    torch::Tensor px = pmc.index({torch::indexing::Slice(), 0});
    torch::Tensor py = pmc.index({torch::indexing::Slice(), 1}); 
    return transform_::Pt(&px, &py); 
}

torch::Tensor pyc::transform::separate::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&px); 
    return transform_::Eta(&px, &py, &pz); 
}

torch::Tensor pyc::transform::combined::Eta(torch::Tensor pmc){
    changedev(&pmc);
    return transform_::Eta(&pmc); 
}

torch::Tensor pyc::transform::separate::Phi(torch::Tensor px, torch::Tensor py){
    changedev(&px); 
    return transform_::Phi(&px, &py); 
}

torch::Tensor pyc::transform::combined::Phi(torch::Tensor pmc){
    torch::Tensor px = pmc.index({torch::indexing::Slice(), 0});
    torch::Tensor py = pmc.index({torch::indexing::Slice(), 1}); 
    changedev(&px); 
    return transform_::Phi(&px, &py); 
}

torch::Tensor pyc::transform::separate::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&px); 
    return transform_::PtEtaPhi(&px, &py, &pz); 
}

torch::Tensor pyc::transform::combined::PtEtaPhi(torch::Tensor pmc){
    changedev(&pmc); 
    return transform_::PtEtaPhi(&pmc); 
}

torch::Tensor pyc::transform::separate::PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&e); 
    return transform_::PtEtaPhiE(&px, &py, &pz, &e); 
}

torch::Tensor pyc::transform::combined::PtEtaPhiE(torch::Tensor pmc){
    changedev(&pmc); 
    return transform_::PtEtaPhiE(&pmc); 
}

torch::Tensor pyc::transform::separate::Px(torch::Tensor pt, torch::Tensor phi){
    changedev(&pt); 
    return transform_::Px(&pt, &phi); 
}

torch::Tensor pyc::transform::combined::Px(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pt = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor phi = pmu.index({torch::indexing::Slice(), 2}); 
    return transform_::Px(&pt, &phi); 
}

torch::Tensor pyc::transform::separate::Py(torch::Tensor pt, torch::Tensor phi){
    changedev(&phi); 
    return transform_::Py(&pt, &phi); 
}

torch::Tensor pyc::transform::combined::Py(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pt = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor phi = pmu.index({torch::indexing::Slice(), 2}); 
    return transform_::Py(&pt, &phi); 
}

torch::Tensor pyc::transform::separate::Pz(torch::Tensor pt, torch::Tensor eta){
    changedev(&pt);
    return transform_::Pz(&pt, &eta); 
}

torch::Tensor pyc::transform::combined::Pz(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pt  = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor eta = pmu.index({torch::indexing::Slice(), 1}); 
    return transform_::Pz(&pt, &eta); 
}

torch::Tensor pyc::transform::separate::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    changedev(&phi); 
    return transform_::PxPyPz(&pt, &eta, &phi); 
}

torch::Tensor pyc::transform::combined::PxPyPz(torch::Tensor pmu){
    changedev(&pmu); 
    return transform_::PxPyPz(&pmu); 
}

torch::Tensor pyc::transform::separate::PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    return transform_::PxPyPzE(&pt, &eta, &phi, &e); 
}

torch::Tensor pyc::transform::combined::PxPyPzE(torch::Tensor pmu){
    changedev(&pmu); 
    return transform_::PxPyPzE(&pmu); 
}

torch::Tensor pyc::physics::cartesian::separate::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&pz); 
    return physics_::P2(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::P2(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::P2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    changedev(&phi); 
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::P2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::P2(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::P2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&pz); 
    return physics_::P(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::P(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::P(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::P(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::P(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::P(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&px);
    return physics_::Beta2(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Beta2(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Beta2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::Beta2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Beta2(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Beta2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&px); 
    return physics_::Beta(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Beta(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Beta(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::Beta(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Beta(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Beta(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&px); 
    return physics_::M2(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::M2(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::M2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::M2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::M2(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::M2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&px); 
    return physics_::M(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::M(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::polar::separate::M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::M(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::M(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Mt2(torch::Tensor pz, torch::Tensor e){
    changedev(&pz); 
    return physics_::Mt2(&pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Mt2(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Mt2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pz = transform_::Pz(&pt, &eta); 
    return physics_::Mt2(&pz, &e); 
}

torch::Tensor pyc::physics::polar::combined::Mt2(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Mt2(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Mt(torch::Tensor pz, torch::Tensor e){
    changedev(&pz); 
    return physics_::Mt(&pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Mt(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Mt(&pmc); 
}

torch::Tensor pyc::physics::polar::separate::Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pz = transform_::Pz(&pt, &eta); 
    return physics_::Mt(&pz, &e); 
}

torch::Tensor pyc::physics::polar::combined::Mt(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Mt(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&px); 
    return physics_::Theta(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::Theta(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Theta(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::Theta(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Theta(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::Theta(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::DeltaR(
        torch::Tensor px1, torch::Tensor px2, 
        torch::Tensor py1, torch::Tensor py2, 
        torch::Tensor pz1, torch::Tensor pz2
){
    changedev(&px1);
    torch::Tensor pmu1 = transform_::PtEtaPhi(&px1, &py1, &pz1); 
    torch::Tensor pmu2 = transform_::PtEtaPhi(&px2, &py2, &pz2); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::physics::cartesian::combined::DeltaR(torch::Tensor pmc1, torch::Tensor pmc2){
    changedev(&pmc1); 
    torch::Tensor pmu1 = transform_::PtEtaPhi(&pmc1); 
    torch::Tensor pmu2 = transform_::PtEtaPhi(&pmc2); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::physics::polar::separate::DeltaR(
        torch::Tensor eta1, torch::Tensor eta2, 
        torch::Tensor phi1, torch::Tensor phi2
){
    changedev(&eta1); 
    return physics_::DeltaR(&eta1, &eta2, &phi1, &phi2); 
}

torch::Tensor pyc::physics::polar::combined::DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){
    changedev(&pmu1); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::operators::Dot(torch::Tensor v1, torch::Tensor v2){
    changedev(&v1); 
    return operators_::Dot(&v1, &v2); 
}

torch::Tensor pyc::operators::CosTheta(torch::Tensor v1, torch::Tensor v2){
    changedev(&v1);
    return operators_::CosTheta(&v1, &v2); 
}

torch::Tensor pyc::operators::SinTheta(torch::Tensor v1, torch::Tensor v2){
    changedev(&v1); 
    return operators_::SinTheta(&v1, &v2); 
}

torch::Tensor pyc::operators::Rx(torch::Tensor angle){
    changedev(&angle); 
    return operators_::Rx(&angle); 
}

torch::Tensor pyc::operators::Ry(torch::Tensor angle){
    changedev(&angle); 
    return operators_::Ry(&angle); 
}

torch::Tensor pyc::operators::Rz(torch::Tensor angle){
    changedev(&angle); 
    return operators_::Rz(&angle); 
}


torch::Tensor pyc::operators::RT(torch::Tensor pmc_b, torch::Tensor pmc_mu){
    changedev(&pmc_b); 
    torch::Tensor phi = pyc::transform::combined::Phi(pmc_mu);
    torch::Tensor theta = physics_::Theta(&pmc_mu); 
    return operators_::RT(&pmc_b, &phi, &theta); 
}

torch::Tensor pyc::operators::CoFactors(torch::Tensor matrix){
    changedev(&matrix); 
    return operators_::CoFactors(&matrix); 
}

torch::Tensor pyc::operators::Determinant(torch::Tensor matrix){
    changedev(&matrix); 
    return operators_::Determinant(&matrix); 
}

std::tuple<torch::Tensor, torch::Tensor> pyc::operators::Inverse(torch::Tensor matrix){
    changedev(&matrix); 
    return operators_::Inverse(&matrix); 
}

torch::Tensor pyc::operators::Cross(torch::Tensor mat1, torch::Tensor mat2){
    changedev(&mat1); 
    return operators_::Cross(&mat1, &mat2); 
}

std::tuple<torch::Tensor, torch::Tensor> pyc::operators::Eigenvalue(torch::Tensor matrix){
    changedev(&matrix); 
    return operators_::Eigenvalue(&matrix); 
}


torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor>* inpt){
    torch::Dict<std::string, torch::Tensor> out;  
    std::map<std::string, torch::Tensor>::iterator itr = inpt -> begin(); 
    for (; itr != inpt -> end(); ++itr){out.insert(itr -> first, itr -> second);}
    return out; 
}

torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor> inpt){
    return pyc::std_to_dict(&inpt); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses){
    changedev(&pmc_b); 
    return pyc::std_to_dict(nusol_::BaseMatrix(&pmc_b, &pmc_mu, &masses)); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy, 
        torch::Tensor masses, torch::Tensor sigma, double null
){
    changedev(&pmc_b); 
    std::map<std::string, torch::Tensor> out = nusol_::Nu(&pmc_b, &pmc_mu, &met_xy, &masses, &sigma, null);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
        torch::Tensor met_xy, torch::Tensor masses, double null
){
    changedev(&pmc_b1); 
    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&pmc_b1, &pmc_b2, &pmc_l1, &pmc_l2, &met_xy, null, &masses);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
        torch::Tensor met_xy, double null, torch::Tensor mass1, torch::Tensor mass2
){
    changedev(&pmc_b1); 
    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&pmc_b1, &pmc_b2, &pmc_l1, &pmc_l2, &met_xy, null, &mass1, &mass2);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::combinatorial(
        torch::Tensor edge_index, torch::Tensor batch, torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy, 
        double mT, double mW, double top_pm, double w_pm, long steps, double null, bool gev
){
    changedev(&edge_index);
    std::map<std::string, torch::Tensor> out;
    out = nusol_::combinatorial(&edge_index, &batch, &pmc, &pid, &met_xy, mT, mW, top_pm, w_pm, steps, null, gev); 
    return pyc::std_to_dict(&out); 
}


torch::Dict<std::string, torch::Tensor> pyc::graph::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor edge_feature
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::edge_aggregation(&edge_index, &prediction, &edge_feature); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::node_aggregation(&edge_index, &prediction, &node_feature); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::unique_aggregation(
        torch::Tensor cluster_map, torch::Tensor feature
){
    changedev(&cluster_map); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::unique_aggregation(&cluster_map, &feature); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = torch::cat({px, py, pz, e}, {-1}); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = torch::cat({px, py, pz, e}, {-1}); 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

