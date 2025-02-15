#include <pyc/tpyc.h>
#include <graph/graph.h>
#include <nusol/nusol.h>
#include <cutils/utils.h>
#include <physics/physics.h>
#include <transform/transform.h>
#include <operators/operators.h>

torch::Tensor pyc::transform::separate::Pt(torch::Tensor px, torch::Tensor py){
    return transform_::Pt(&px, &py); 
}

torch::Tensor pyc::transform::combined::Pt(torch::Tensor pmc){
    torch::Tensor px = pmc.index({torch::indexing::Slice(), 0});
    torch::Tensor py = pmc.index({torch::indexing::Slice(), 1}); 
    return transform_::Pt(&px, &py); 
}

torch::Tensor pyc::transform::separate::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return transform_::Eta(&px, &py, &pz); 
}

torch::Tensor pyc::transform::combined::Eta(torch::Tensor pmc){
    return transform_::Eta(&pmc); 
}

torch::Tensor pyc::transform::separate::Phi(torch::Tensor px, torch::Tensor py){
    return transform_::Phi(&px, &py); 
}

torch::Tensor pyc::transform::combined::Phi(torch::Tensor pmc){
    torch::Tensor px = pmc.index({torch::indexing::Slice(), 0});
    torch::Tensor py = pmc.index({torch::indexing::Slice(), 1}); 
    return transform_::Phi(&px, &py); 
}

torch::Tensor pyc::transform::separate::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return transform_::PtEtaPhi(&px, &py, &pz); 
}

torch::Tensor pyc::transform::combined::PtEtaPhi(torch::Tensor pmc){
    return transform_::PtEtaPhi(&pmc); 
}

torch::Tensor pyc::transform::separate::PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return transform_::PtEtaPhiE(&px, &py, &pz, &e); 
}

torch::Tensor pyc::transform::combined::PtEtaPhiE(torch::Tensor pmc){
    return transform_::PtEtaPhiE(&pmc); 
}

torch::Tensor pyc::transform::separate::Px(torch::Tensor pt, torch::Tensor phi){
    return transform_::Px(&pt, &phi); 
}

torch::Tensor pyc::transform::combined::Px(torch::Tensor pmu){
    torch::Tensor pt = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor phi = pmu.index({torch::indexing::Slice(), 2}); 
    return transform_::Px(&pt, &phi); 
}

torch::Tensor pyc::transform::separate::Py(torch::Tensor pt, torch::Tensor phi){
    return transform_::Py(&pt, &phi); 
}

torch::Tensor pyc::transform::combined::Py(torch::Tensor pmu){
    torch::Tensor pt = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor phi = pmu.index({torch::indexing::Slice(), 2}); 
    return transform_::Py(&pt, &phi); 
}

torch::Tensor pyc::transform::separate::Pz(torch::Tensor pt, torch::Tensor eta){
    return transform_::Pz(&pt, &eta); 
}

torch::Tensor pyc::transform::combined::Pz(torch::Tensor pmu){
    torch::Tensor pt  = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor eta = pmu.index({torch::indexing::Slice(), 1}); 
    return transform_::Pz(&pt, &eta); 
}

torch::Tensor pyc::transform::separate::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    return transform_::PxPyPz(&pt, &eta, &phi); 
}

torch::Tensor pyc::transform::combined::PxPyPz(torch::Tensor pmu){
    return transform_::PxPyPz(&pmu); 
}

torch::Tensor pyc::transform::separate::PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    return transform_::PxPyPzE(&pt, &eta, &phi, &e); 
}

torch::Tensor pyc::transform::combined::PxPyPzE(torch::Tensor pmu){
    return transform_::PxPyPzE(&pmu); 
}

torch::Tensor pyc::physics::cartesian::separate::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return physics_::P2(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::P2(torch::Tensor pmc){
    return physics_::P2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::P2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::P2(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::P2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return physics_::P(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::P(torch::Tensor pmc){
    return physics_::P(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::P(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::P(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::P(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return physics_::Beta2(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Beta2(torch::Tensor pmc){
    return physics_::Beta2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::Beta2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Beta2(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Beta2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return physics_::Beta(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Beta(torch::Tensor pmc){
    return physics_::Beta(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::Beta(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Beta(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Beta(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return physics_::M2(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::M2(torch::Tensor pmc){
    return physics_::M2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::M2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::M2(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::M2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return physics_::M(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::M(torch::Tensor pmc){
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::polar::separate::M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::M(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Mt2(torch::Tensor pz, torch::Tensor e){
    return physics_::Mt2(&pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Mt2(torch::Tensor pmc){
    return physics_::Mt2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){
    torch::Tensor pz = transform_::Pz(&pt, &eta); 
    return physics_::Mt2(&pz, &e); 
}

torch::Tensor pyc::physics::polar::combined::Mt2(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Mt2(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Mt(torch::Tensor pz, torch::Tensor e){
    return physics_::Mt(&pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Mt(torch::Tensor pmc){
    return physics_::Mt(&pmc); 
}

torch::Tensor pyc::physics::polar::separate::Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){
    torch::Tensor pz = transform_::Pz(&pt, &eta); 
    return physics_::Mt(&pz, &e); 
}

torch::Tensor pyc::physics::polar::combined::Mt(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Mt(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return physics_::Theta(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::Theta(torch::Tensor pmc){
    return physics_::Theta(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::Theta(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Theta(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::Theta(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::DeltaR(
        torch::Tensor px1, torch::Tensor px2, 
        torch::Tensor py1, torch::Tensor py2, 
        torch::Tensor pz1, torch::Tensor pz2
){
    torch::Tensor pmu1 = transform_::PtEtaPhi(&px1, &py1, &pz1); 
    torch::Tensor pmu2 = transform_::PtEtaPhi(&px2, &py2, &pz2); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::physics::cartesian::combined::DeltaR(torch::Tensor pmc1, torch::Tensor pmc2){
    torch::Tensor pmu1 = transform_::PtEtaPhi(&pmc1); 
    torch::Tensor pmu2 = transform_::PtEtaPhi(&pmc2); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::physics::polar::separate::DeltaR(
        torch::Tensor eta1, torch::Tensor eta2, 
        torch::Tensor phi1, torch::Tensor phi2
){
    return physics_::DeltaR(&eta1, &eta2, &phi1, &phi2); 
}

torch::Tensor pyc::physics::polar::combined::DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::operators::Dot(torch::Tensor v1, torch::Tensor v2){
    return operators_::Dot(&v1, &v2); 
}

torch::Tensor pyc::operators::CosTheta(torch::Tensor v1, torch::Tensor v2){
    return operators_::CosTheta(&v1, &v2); 
}

torch::Tensor pyc::operators::SinTheta(torch::Tensor v1, torch::Tensor v2){
    return operators_::SinTheta(&v1, &v2); 
}

torch::Tensor pyc::operators::Rx(torch::Tensor angle){
    return operators_::Rx(&angle); 
}

torch::Tensor pyc::operators::Ry(torch::Tensor angle){
    return operators_::Ry(&angle); 
}

torch::Tensor pyc::operators::Rz(torch::Tensor angle){
    return operators_::Rz(&angle); 
}


torch::Tensor pyc::operators::CoFactors(torch::Tensor matrix){
    return operators_::CoFactors(&matrix); 
}

torch::Tensor pyc::operators::Determinant(torch::Tensor matrix){
    return operators_::Determinant(&matrix); 
}

torch::Tensor pyc::operators::Inverse(torch::Tensor matrix){
    return operators_::Inverse(&matrix); 
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

torch::Tensor pyc::nusol::BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses){
    return nusol_::Hperp(&pmc_b, &pmc_mu, &masses); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy, 
        torch::Tensor masses, torch::Tensor sigma, double null
){
    std::map<std::string, torch::Tensor> out = nusol_::Nu(&pmc_b, &pmc_mu, &met_xy, &masses, &sigma, null);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
        torch::Tensor met_xy, torch::Tensor masses, double null
){
    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&pmc_b1, &pmc_b2, &pmc_l1, &pmc_l2, &met_xy, null, &masses);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::combinatorial(
        torch::Tensor edge_index, torch::Tensor batch , torch::Tensor pmc, 
        torch::Tensor pid       , torch::Tensor met_xy, 
        double mT, double mW, double top_pm, double w_pm, long steps, double null, bool gev
){
    std::map<std::string, torch::Tensor> out; 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::unique_aggregation(
        torch::Tensor cluster_map, torch::Tensor features
){
    std::map<std::string, torch::Tensor> out; 
    return pyc::std_to_dict(&out); 
}
 
torch::Dict<std::string, torch::Tensor> pyc::graph::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor edge_feature
){
    std::map<std::string, torch::Tensor> out; 
    out = graph_::edge_aggregation(&edge_index, &prediction, &edge_feature); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
){
    std::map<std::string, torch::Tensor> out; 
    out = graph_::node_aggregation(&edge_index, &prediction, &node_feature); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
){
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
){
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
){
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
){
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
){
    std::map<std::string, torch::Tensor> out; 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
){
    std::map<std::string, torch::Tensor> out; 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
){
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = torch::cat({px, py, pz, e}, {-1}); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
){
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = torch::cat({px, py, pz, e}, {-1}); 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

