# distutils: language=c++
# cython: language_level=3
from AnalysisG.core.tools cimport *

cdef void get_training(PageRankMetric vl, dict data, dict meta):
    cdef int epoch = meta[b"epoch"]
    cdef int kfold = meta[b"kfold"]
    cdef string model = meta[b"model_name"]
    cdef string fname = meta[b"filename"]
    cdef string mode = b""
    cdef bool stat = False
    

    cdef collector* cl = vl.cl
    if cl.add_meta(fname, model, kfold, epoch): 
        for i in open(env(fname) + ".txt", "r").readlines():
            if "<file-mapping>" in i: continue
            if "<file-statistics>" in i: stat = True; continue
            if "training"   in i: mode = b"training"; continue
            if "validation" in i: mode = b"validation"; continue
            if "evaluation" in i: mode = b"evaluation"; continue
            xl = "::".join(i.split("::")[:-1])
            cl.add_file_map(enc(xl), int(i.split("::")[-1]), mode, model, kfold, epoch, stat)

    cl.set_index(meta[b"index"])
    process(cl, b"training", data, model, kfold, epoch)

cdef void get_validation(PageRankMetric vl, dict data, dict meta):
    cdef int epoch = meta[b"epoch"]
    cdef int kfold = meta[b"kfold"]
    cdef string model = meta[b"model_name"]
    cdef string fname = meta[b"filename"]
    cdef string mode = b""
    cdef bool stat = False
 
    cdef collector* cl = vl.cl
    if cl.add_meta(fname, model, kfold, epoch):
        for i in open(env(fname) + ".txt", "r").readlines():
            if "<file-mapping>" in i: continue
            if "<file-statistics>" in i: stat = True; continue
            if "training"   in i: mode = b"training"; continue
            if "validation" in i: mode = b"validation"; continue
            if "evaluation" in i: mode = b"evaluation"; continue
            xl = "::".join(i.split("::")[:-1])
            cl.add_file_map(enc(xl), int(i.split("::")[-1]), mode, model, kfold, epoch, stat)

    cl.set_index(meta[b"index"])
    process(cl, b"validation", data, model, kfold, epoch)


cdef void get_evaluation(PageRankMetric vl, dict data, dict meta):
    cdef int epoch = meta[b"epoch"]
    cdef int kfold = meta[b"kfold"]
    cdef string model = meta[b"model_name"]
    cdef string fname = meta[b"filename"]
    cdef string mode = b""
    cdef bool stat = False
 
    cdef collector* cl = vl.cl
    if cl.add_meta(fname, model, kfold, epoch):
        for i in open(env(fname) + ".txt", "r").readlines():
            if "<file-mapping>" in i: continue
            if "<file-statistics>" in i: stat = True; continue
            if "training"   in i: mode = b"training"; continue
            if "validation" in i: mode = b"validation"; continue
            if "evaluation" in i: mode = b"evaluation"; continue
            xl = "::".join(i.split("::")[:-1])
            cl.add_file_map(enc(xl), int(i.split("::")[-1]), mode, model, kfold, epoch, stat)

    cl.set_index(meta[b"index"])
    process(cl, b"evaluation", data, model, kfold, epoch)










cdef class PageRankMetric(MetricTemplate):
    def __cinit__(self):

        keys = [
            "top_truth_pt", "top_truth_eta", "top_truth_phi", "top_truth_energy",
            "top_truth_px", "top_truth_py" , "top_truth_pz" , "top_truth_mass",
            "top_truth_num_nodes", 

            "top_pr_reco_pt", "top_pr_reco_eta", "top_pr_reco_phi", "top_pr_reco_energy", 
            "top_pr_reco_px", "top_pr_reco_py" , "top_pr_reco_pz" , "top_pr_reco_mass", 
            "top_pr_reco_pagerank", "top_pr_reco_num_nodes", 

            "top_nom_reco_pt", "top_nom_reco_eta", "top_nom_reco_phi", "top_nom_reco_energy",   
            "top_nom_reco_px", "top_nom_reco_py" , "top_nom_reco_pz" , "top_nom_reco_mass",
            "top_nom_reco_score", "top_nom_reco_num_nodes", "process_mapping"  
        ]

        self.root_leaves = {"event_training" : keys, "event_validation" : keys, "event_evaluation" : keys}
        self.root_fx = {"event_training" : get_training, "event_validation" : get_validation, "event_evaluation" : get_evaluation}

        self.cl  = new collector()
        self.mtx = new pagerank_metric()
        self.mtr = <pagerank_metric*>(self.mtx)


    
