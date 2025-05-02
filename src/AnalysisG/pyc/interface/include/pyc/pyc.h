#ifndef PYC_H
#define PYC_H

#include <tuple>
#include <torch/all.h> 
#include <templates/particle_template.h>

class neutrino: public particle_template
{
    public:
        neutrino(); 
        neutrino(double px, double py, double pz);

        virtual ~neutrino(); 
        double min = 0; 
        long l_idx = -1; 
        long b_idx = -1; 

        particle_template* bquark = nullptr; 
        particle_template* lepton = nullptr; 
}; 

namespace pyc {
    torch::Dict<std::string, torch::Tensor> std_to_dict(std::map<std::string, torch::Tensor>* inpt); 
    torch::Dict<std::string, torch::Tensor> std_to_dict(std::map<std::string, torch::Tensor> inpt); 
    torch::Tensor tensorize(std::vector<std::vector<double>>* inpt);
    torch::Tensor tensorize(std::vector<std::vector<long>>* inpt);
    torch::Tensor tensorize(std::vector<double>* inpt);
    torch::Tensor tensorize(std::vector<long>* inpt); 

    template <typename g>
    std::vector<double> as_pmc(g* p){return {p -> px, p -> py, p -> pz, p -> e};}

    template <typename g>
    std::vector<std::vector<double>> to_pmc(std::vector<g*>* p){
        std::vector<std::vector<double>> atx; 
        for (size_t x(0); x < p -> size(); ++x){atx.push_back(pyc::as_pmc((*p)[x]));}
        return atx; 
    }

    namespace transform {
        namespace separate {
            torch::Tensor Pt(torch::Tensor px, torch::Tensor py); 
            torch::Tensor Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
            torch::Tensor Phi(torch::Tensor px, torch::Tensor py); 
            torch::Tensor PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
            torch::Tensor PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

            torch::Tensor Px(torch::Tensor pt, torch::Tensor phi); 
            torch::Tensor Py(torch::Tensor pt, torch::Tensor phi); 
            torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta); 
            torch::Tensor PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
            torch::Tensor PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
        } 
        namespace combined {    
            torch::Tensor Pt(torch::Tensor pmc); 
            torch::Tensor Eta(torch::Tensor pmc); 
            torch::Tensor Phi(torch::Tensor pmc); 
            torch::Tensor PtEtaPhi(torch::Tensor pmc); 
            torch::Tensor PtEtaPhiE(torch::Tensor pmc); 

            torch::Tensor Px(torch::Tensor pmu); 
            torch::Tensor Py(torch::Tensor pmu); 
            torch::Tensor Pz(torch::Tensor pmu); 
            torch::Tensor PxPyPz(torch::Tensor pmu); 
            torch::Tensor PxPyPzE(torch::Tensor pmu); 
        }
    }

    namespace physics {
        namespace cartesian {
            namespace separate {
                torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
                torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
                torch::Tensor Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor Mt2(torch::Tensor pz, torch::Tensor e);
                torch::Tensor Mt(torch::Tensor pz, torch::Tensor e);
                torch::Tensor Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
                torch::Tensor DeltaR(torch::Tensor px1, torch::Tensor px2, torch::Tensor py1, torch::Tensor py2, torch::Tensor pz1, torch::Tensor pz2);
            }
            namespace combined {
                torch::Tensor P2(torch::Tensor pmc); 
                torch::Tensor P(torch::Tensor pmc); 
                torch::Tensor Beta2(torch::Tensor pmc); 
                torch::Tensor Beta(torch::Tensor pmc); 
                torch::Tensor M2(torch::Tensor pmc);
                torch::Tensor M(torch::Tensor pmc);
                torch::Tensor Mt2(torch::Tensor pmc);
                torch::Tensor Mt(torch::Tensor pmc);
                torch::Tensor Theta(torch::Tensor pmc); 
                torch::Tensor DeltaR(torch::Tensor pmc1, torch::Tensor pmc2); 
            }
        }
        namespace polar {
            namespace separate {
                torch::Tensor P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
                torch::Tensor P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
                torch::Tensor Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
                torch::Tensor Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
                torch::Tensor M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
                torch::Tensor M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
                torch::Tensor Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e); 
                torch::Tensor Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e); 
                torch::Tensor Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
                torch::Tensor DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2); 
            }
            namespace combined {
                torch::Tensor P2(torch::Tensor pmu); 
                torch::Tensor P(torch::Tensor pmu); 
                torch::Tensor Beta2(torch::Tensor pmu); 
                torch::Tensor Beta(torch::Tensor pmu); 
                torch::Tensor M2(torch::Tensor pmu); 
                torch::Tensor M(torch::Tensor pmu); 
                torch::Tensor Mt2(torch::Tensor pmu); 
                torch::Tensor Mt(torch::Tensor pmu); 
                torch::Tensor Theta(torch::Tensor pmu); 
                torch::Tensor DeltaR(torch::Tensor pmu1, torch::Tensor pmu2); 
            }
        }
    }

    namespace operators {
        torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2); 
        torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2); 
        torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2);
        torch::Tensor Rx(torch::Tensor angle); 
        torch::Tensor Ry(torch::Tensor angle); 
        torch::Tensor Rz(torch::Tensor angle); 
        torch::Tensor RT(torch::Tensor pmc_b, torch::Tensor pmc_mu); 

        torch::Tensor CoFactors(torch::Tensor matrix);
        torch::Tensor Determinant(torch::Tensor matrix); 
        std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor matrix); 
        std::tuple<torch::Tensor, torch::Tensor> Eigenvalue(torch::Tensor matrix); 
        torch::Tensor Cross(torch::Tensor mat1, torch::Tensor mat2); 
    }

    namespace nusol {
        torch::Dict<std::string, torch::Tensor> BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses); 
        torch::Dict<std::string, torch::Tensor> Nu(
               torch::Tensor pmc_b , torch::Tensor pmc_mu, torch::Tensor met_xy, 
               torch::Tensor masses, torch::Tensor  sigma, double null = 10e-10
        ); 

        torch::Dict<std::string, torch::Tensor> NuNu(
               torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
               torch::Tensor met_xy, torch::Tensor masses, double null = 10e-10, const double step = 1e-9, 
               const double tolerance = 1e-6, const unsigned int timeout = 1000
        ); 


        torch::Dict<std::string, torch::Tensor> NuNu(
               torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
               torch::Tensor met_xy, double null, torch::Tensor mass1, torch::Tensor mass2, const double step = 1e-9, 
               const double tolerance = 1e-6, const unsigned int timeout = 1000
        ); 


        std::vector<std::pair<neutrino*, neutrino*>> NuNu(
               std::vector<std::vector<double>>* pmc_b1, std::vector<std::vector<double>>* pmc_b2, 
               std::vector<std::vector<double>>* pmc_l1, std::vector<std::vector<double>>* pmc_l2, 
                           std::vector<double>*    met,             std::vector<double>*  phi, 
               std::vector<std::vector<double>>* mass1, std::vector<std::vector<double>>* mass2,  
               std::string dev, const double null, const double step, const double tolerance, const unsigned int timeout
        ); 

        template <typename b, typename l>
        std::vector<std::pair<neutrino*, neutrino*>> NuNu(
               std::vector<b*> bquark1, std::vector<b*> bquark2, 
               std::vector<l*> lepton1, std::vector<l*> lepton2, 
               std::vector<double> met_, std::vector<double> phi_,
               std::vector<std::vector<double>> mass1, std::vector<std::vector<double>> mass2, 
               std::string dev, double null, const double step, const double tolerance, const unsigned int timeout
        ){

            std::vector<std::vector<double>> b1 = pyc::to_pmc(&bquark1); 
            std::vector<std::vector<double>> b2 = pyc::to_pmc(&bquark2); 
            std::vector<std::vector<double>> l1 = pyc::to_pmc(&lepton1); 
            std::vector<std::vector<double>> l2 = pyc::to_pmc(&lepton2); 

            std::vector<std::pair<neutrino*, neutrino*>> out; 
            out = pyc::nusol::NuNu(&b1, &b2, &l1, &l2, &met_, &phi_, &mass1, &mass2, dev, null, step, tolerance, timeout); 

            for (size_t x(0); x < out.size(); ++x){
                neutrino* nu1 = std::get<0>(out[x]); 
                nu1 -> bquark = new particle_template(bquark1[nu1 -> b_idx]); 
                nu1 -> lepton = new particle_template(lepton1[nu1 -> l_idx]); 

                neutrino* nu2 = std::get<1>(out[x]); 
                nu2 -> bquark = new particle_template(bquark2[nu2 -> b_idx]); 
                nu2 -> lepton = new particle_template(lepton2[nu2 -> l_idx]); 
            }
            return out; 
        }


        torch::Dict<std::string, torch::Tensor> combinatorial(
               torch::Tensor edge_index, torch::Tensor batch , torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy, 
               double mT  = 172.62*1000 , double mW = 80.385*1000, double null = 1e-10, double perturb = 1e-3, 
               long steps = 100, bool gev = false
        ); 

        std::vector<std::pair<neutrino*, neutrino*>> combinatorial(
                std::vector<double>* met_, std::vector<double>* phi_, std::vector<std::vector<double>>* pmc, 
                std::vector<long>* bth, std::vector<long>* is_b, std::vector<long>* is_l, std::string dev,
                double mT, double mW, double null, double perturb, long steps
        ); 

        std::vector<std::pair<neutrino*, neutrino*>> combinatorial(
               std::vector<double> met_, std::vector<double> phi_, 
               std::vector<std::vector<particle_template*>> particles,
               std::string dev, double mT, double mW, double null, double perturb, long steps
        );
    }

    namespace graph {
        torch::Dict<std::string, torch::Tensor> edge_aggregation(
            torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
        ); 

        torch::Dict<std::string, torch::Tensor> node_aggregation(
            torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
        ); 

        torch::Dict<std::string, torch::Tensor> unique_aggregation(
                torch::Tensor cluster_map, torch::Tensor features
        ); 

        torch::Dict<std::string, torch::Tensor> PageRank(
                torch::Tensor edge_index, torch::Tensor edge_scores, 
                double alpha = 0.85, double threshold = 0.5, double norm_low = 1e-6, long timeout = 1e6, long num_cls = 2
        );

        torch::Dict<std::string, torch::Tensor> PageRankReconstruction(
                torch::Tensor edge_index, torch::Tensor edge_scores, torch::Tensor pmc, 
                double alpha = 0.85, double threshold = 0.5, double norm_low = 1e-6, long timeout = 1e6, long num_cls = 2
        ); 
         
        namespace polar {
            torch::Dict<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
            );

            torch::Dict<std::string, torch::Tensor> node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
            ); 

            torch::Dict<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, 
                torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
            ); 

            torch::Dict<std::string, torch::Tensor> node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, 
                torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
            );  
        }

        namespace cartesian {
            torch::Dict<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
            );

            torch::Dict<std::string, torch::Tensor> node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
            ); 

            torch::Dict<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, 
                torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
            ); 

            torch::Dict<std::string, torch::Tensor> node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, 
                torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
            );  
        }
    }
}

#endif
