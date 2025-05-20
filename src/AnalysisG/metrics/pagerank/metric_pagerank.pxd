# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.metric_template cimport *
from AnalysisG.core.tools cimport *

cdef extern from "<metrics/pagerank.h>":
    cdef cppclass pagerank_metric(metric_template):
        pagerank_metric() except+

        float alpha 
        float norm_lim 
        unsigned long int max_itr
  

    cdef struct kinematic_t:
        double px
        double py
        double pz
        double mass
        
        double pt     
        double eta
        double phi
        double energy

        int num_nodes
        double score

    cdef cppclass collector(tools):
        void set_index(long idx_) except+

        kinematic_t create_kinematic(
                double px, double py,  double pz,  double mass, 
                double pt, double eta, double phi, double energy, 
                int num_nodes, double score
        ) except+

        void add_truth(kinematic_t* p, string mode, string model, int kfold, int epoch)
        void add_pagerank(kinematic_t* p, string mode, string model, int kfold, int epoch)
        void add_nominal(kinematic_t* p, string mode, string model, int kfold, int epoch)
        void add_process(int prc, string mode, string model, int kfold, int epoch)
        bool add_meta(string data, string model, int kfold, int epoch)
        void add_file_map(string fname, long idx_, string mode, string model, int kfold, int epoch, bool stat)


cdef inline void process(collector* cl, string mode, dict data, string model, int kfold, int epoch):
    cdef int ix
    cdef kinematic_t kx

    cdef vector[double] tru_pt  = data[b"event_" + mode + b".top_truth_pt.top_truth_pt"]
    cdef vector[double] tru_eta = data[b"event_" + mode + b".top_truth_eta.top_truth_eta"]
    cdef vector[double] tru_phi = data[b"event_" + mode + b".top_truth_phi.top_truth_phi"]
    cdef vector[double] tru_enr = data[b"event_" + mode + b".top_truth_energy.top_truth_energy"]
    cdef vector[double] tru_px  = data[b"event_" + mode + b".top_truth_px.top_truth_px"]
    cdef vector[double] tru_py  = data[b"event_" + mode + b".top_truth_py.top_truth_py"]
    cdef vector[double] tru_pz  = data[b"event_" + mode + b".top_truth_pz.top_truth_pz"]
    cdef vector[double] tru_mss = data[b"event_" + mode + b".top_truth_mass.top_truth_mass"]
    cdef vector[int]    tru_num = data[b"event_" + mode + b".top_truth_num_nodes.top_truth_num_nodes"]
    for ix in range(tru_num.size()):
        kx = cl.create_kinematic(tru_px[ix], tru_py[ix], tru_pz[ix], tru_mss[ix], tru_pt[ix], tru_eta[ix], tru_phi[ix], tru_enr[ix], tru_num[ix], -1); 
        cl.add_truth(&kx, mode, model, kfold, epoch)

    cdef vector[double] pr_pt  = data[b"event_" + mode + b".top_pr_reco_pt.top_pr_reco_pt"]
    cdef vector[double] pr_eta = data[b"event_" + mode + b".top_pr_reco_eta.top_pr_reco_eta"]
    cdef vector[double] pr_phi = data[b"event_" + mode + b".top_pr_reco_phi.top_pr_reco_phi"]
    cdef vector[double] pr_enr = data[b"event_" + mode + b".top_pr_reco_energy.top_pr_reco_energy"]
    cdef vector[double] pr_px  = data[b"event_" + mode + b".top_pr_reco_px.top_pr_reco_px"]
    cdef vector[double] pr_py  = data[b"event_" + mode + b".top_pr_reco_py.top_pr_reco_py"]
    cdef vector[double] pr_pz  = data[b"event_" + mode + b".top_pr_reco_pz.top_pr_reco_pz"]
    cdef vector[double] pr_mss = data[b"event_" + mode + b".top_pr_reco_mass.top_pr_reco_mass"]
    cdef vector[double] pr_sc  = data[b"event_" + mode + b".top_pr_reco_pagerank.top_pr_reco_pagerank"]
    cdef vector[int]    pr_num = data[b"event_" + mode + b".top_pr_reco_num_nodes.top_pr_reco_num_nodes"]
    for ix in range(pr_num.size()):
        kx = cl.create_kinematic(pr_px[ix], pr_py[ix], pr_pz[ix], pr_mss[ix], pr_pt[ix], pr_eta[ix], pr_phi[ix], pr_enr[ix], pr_num[ix], pr_sc[ix]); 
        cl.add_pagerank(&kx, mode, model, kfold, epoch)

    cdef vector[double] nm_pt  = data[b"event_" + mode + b".top_nom_reco_pt.top_nom_reco_pt"]
    cdef vector[double] nm_eta = data[b"event_" + mode + b".top_nom_reco_eta.top_nom_reco_eta"]
    cdef vector[double] nm_phi = data[b"event_" + mode + b".top_nom_reco_phi.top_nom_reco_phi"]
    cdef vector[double] nm_enr = data[b"event_" + mode + b".top_nom_reco_energy.top_nom_reco_energy"]
    cdef vector[double] nm_px  = data[b"event_" + mode + b".top_nom_reco_px.top_nom_reco_px"]
    cdef vector[double] nm_py  = data[b"event_" + mode + b".top_nom_reco_py.top_nom_reco_py"]
    cdef vector[double] nm_pz  = data[b"event_" + mode + b".top_nom_reco_pz.top_nom_reco_pz"]
    cdef vector[double] nm_mss = data[b"event_" + mode + b".top_nom_reco_mass.top_nom_reco_mass"]
    cdef vector[double] nm_sc  = data[b"event_" + mode + b".top_nom_reco_score.top_nom_reco_score"]
    cdef vector[int]    nm_num = data[b"event_" + mode + b".top_nom_reco_num_nodes.top_nom_reco_num_nodes"]
    for ix in range(nm_num.size()):
        kx = cl.create_kinematic(nm_px[ix], nm_py[ix], nm_pz[ix], nm_mss[ix], nm_pt[ix], nm_eta[ix], nm_phi[ix], nm_enr[ix], nm_num[ix], nm_sc[ix]); 
        cl.add_nominal(&kx, mode, model, kfold, epoch)
    cl.add_process(data[b"event_" + mode + b".process_mapping.process_mapping"], mode, model, kfold, epoch)
    

cdef class PageRankMetric(MetricTemplate):
    cdef pagerank_metric* mtr
    cdef collector* cl


