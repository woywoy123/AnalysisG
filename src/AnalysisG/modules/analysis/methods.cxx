#include <AnalysisG/analysis.h>
#include <ROOT/RDataFrame.hxx>
#include <tools/vector_cast.h>
#include <TFile.h>
#include <TTree.h>

void analysis::initialize_loop(
        optimizer* op, int k, model_template* model, 
        optimizer_params_t* config, model_report** rep
){
    ROOT::EnableImplicitMT(); 
    model_settings_t settings; 
    model -> clone_settings(&settings); 

    #ifdef PYC_CUDA
    c10::cuda::set_device(model -> m_option -> get_device()); 
    #endif 
    
    model_template* mk = model -> clone(); 
    std::string pth = model -> model_checkpoint_path; 

    mk -> import_settings(&settings); 
    mk -> set_optimizer(config -> optimizer); 
    mk -> initialize(config); 

    mk -> epoch = 0; 
    mk -> kfold = k+1; 
    for (int ep(0); ep < op -> m_settings.epochs; ++ep){

         // check if the next epoch has a file i+2;
        std::string pth_ = pth + "state/epoch-" + std::to_string(ep+1) + "/";  
        pth_ += "kfold-" + std::to_string(k+1) + "_model.pt"; 

        if (op -> m_settings.continue_training && op -> is_file(pth_)){continue;}
        if (!op -> m_settings.continue_training){break;} 
        mk -> epoch = ep;
        mk -> restore_state(); 
        break; 
    }

    std::vector<graph_t*> rnd = op -> loader -> get_random(1); 
    mk -> shush = true; 
    mk -> check_features(rnd[0]);
    op -> kfold_sessions[k] = mk;
    model_report* mr = op -> metric -> register_model(mk, k); 
    op -> reports[mr -> run_name + std::to_string(mr -> k)] = mr; 
    (*rep) = mr; 
    op -> launch_model(k);
}

int analysis::add_content(
        std::map<std::string, torch::Tensor*>* data, 
        std::vector<variable_t>* content, int index, 
        std::string prefx, TTree* tt
){
    std::map<std::string, torch::Tensor*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr, ++index){
        if (!tt){content -> push_back(variable_t());}
        std::string name = prefx + itr -> first; 
        (*content)[index].process(itr -> second, &name, tt); 
    }
    return index; 
}

void analysis::add_content(
        std::map<std::string, torch::Tensor*>* data, std::vector<std::vector<torch::Tensor>>* buff, 
        torch::Tensor* edge, torch::Tensor* node, torch::Tensor* batch, std::vector<long> mask
){
    torch::Tensor edge_i = edge -> view({-1}); 
    torch::Tensor node_i = node -> view({-1}); 
    torch::Tensor batch_i = batch -> view({-1}); 

    long ei = edge_i.size({0});
    long ni = node_i.size({0}); 
    long bi = batch_i.size({0}); 
    std::map<std::string, torch::Tensor*>::iterator itr; 
    for (itr = data -> begin(); itr != data -> end(); ++itr){
        torch::Tensor* tn = itr -> second;
        torch::Tensor* idx = nullptr;  
        if      (tn -> size({0}) == ei){idx = &edge_i;}
        else if (tn -> size({0}) == ni){idx = &node_i;}
        else if (tn -> size({0}) == bi){idx = &batch_i;}
        else {continue;}
        for (size_t x(0); x < mask.size(); ++x){(*buff)[x].push_back(tn -> index({(*idx) == mask[x]}));} 
    }
}

void analysis::execution(
        model_template* mdx, model_settings_t mds, std::vector<graph_t*>* data, size_t* prg,
        std::string output, std::vector<variable_t>* content, std::string* msg
){
    (*prg) = 1; 
    size_t ds = 0; 
    for (size_t x(0); x < data -> size(); ++x){ds += (*data)[x] -> batched_events.size();}

    TFile* tmp = TFile::Open(output.c_str(), "READ");
    if (tmp){
        TDirectory* dir = gDirectory; 
        for (TObject* key : *dir -> GetListOfKeys()){
            std::string name = std::string(key -> GetName()); 
            if (name != mds.tree_name){continue;}
            TTree* tx = tmp -> Get<TTree>(key -> GetName());
            size_t l = tx -> GetEntries();
            if (l != ds){break;}
            delete content; content = nullptr; 
            delete tmp; tmp = nullptr; 
            (*msg) = "\033[1;33m (Already Completed) " + (*msg) + "\033[0m"; 
            *prg = ds; 
            return;
        }
        delete tmp; 
    }

    model_template* md = mdx -> clone(); 
    md -> import_settings(&mds); 
    if(!md -> restore_state()){
        (*msg) = "\033[1;31m (Missing Model) " + (*msg) + "\033[0m"; 
        std::this_thread::sleep_for(std::chrono::seconds(1));
        delete content; content = nullptr; 
        delete md; md = nullptr; 
        (*prg) = ds; 
        return; 
    }
    md -> shush = true; 

    (*msg) = "\033[1;32m (Processing) " + (*msg) + "\033[0m";

    TFile* f = TFile::Open(output.c_str(), "RECREATE"); 
    f -> TestBit(TFile::kRecovered);  
    TTree* t = new TTree(mds.tree_name.c_str(), "data");
    t -> SetCacheSize(10000000U); 

    (*prg) = 0; 
    torch::AutoGradMode grd(false); 
    std::map<std::string, torch::Tensor*> addhoc;
    for (size_t x(0); x < data -> size(); ++x){
        graph_t* gr = (*data)[x]; 
        md -> forward(gr, false);  

        std::vector<std::vector<torch::Tensor>> bf = {};
        std::vector<long> batch_i = gr -> batched_events; 
        bf.assign(batch_i.size(), {});

        torch::Tensor* bt = gr -> get_batched_events(md); 
        torch::Tensor* nb = gr -> get_batch_index(md); 
        torch::Tensor ei  = gr -> get_edge_index(md) -> transpose(0, 1); 
        torch::Tensor ex  = nb -> index({ei.index({torch::indexing::Slice(), 0})});
        addhoc["edge_index"] = &ei; 
        addhoc[mds.weight_name] = (*data)[x] -> get_event_weight(md); 

        if (!x){
            int index = 0; 
            // --- Scan the inputs
            index = add_content(&md -> m_i_graph, content, index, "g_i_", t); 
            index = add_content(&md -> m_i_node , content, index, "n_i_", t); 
            index = add_content(&md -> m_i_edge , content, index, "e_i_", t); 

            // --- Scan the outputs
            index = add_content(&md -> m_p_graph, content, index, "g_o_", t); 
            index = add_content(&md -> m_p_node , content, index, "n_o_", t); 
            index = add_content(&md -> m_p_edge , content, index, "e_o_", t); 

            // --- add additional content
            index = add_content(&addhoc, content, index, "", t); 
            index = add_content(&md -> m_p_undef, content, index, "extra_", t); 

            std::string bt = ""; 
            for (size_t i(0); i < content -> size(); ++i){
                if (!(*content)[i].failed_branch){continue;}
                std::string f = (*content)[i].variable_name + (*content)[i].scan_buffer(); 
                bt += (!bt.size()) ? " > " + f : f; 
            }
            if (bt.size()){
                (*msg) = "\033[1;31m (Failed to Initialize Branch) " + bt + "\033[0m";
                std::this_thread::sleep_for(std::chrono::seconds(1));
                break; 
            }

        }

        add_content(&md -> m_i_graph, &bf, bt, nb, &ex, batch_i); 
        add_content(&md -> m_i_node,  &bf, bt, nb, &ex, batch_i); 
        add_content(&md -> m_i_edge,  &bf, bt, nb, &ex, batch_i); 

        // --- Scan the outputs
        add_content(&md -> m_p_graph, &bf, bt, nb, &ex, batch_i); 
        add_content(&md -> m_p_node,  &bf, bt, nb, &ex, batch_i); 
        add_content(&md -> m_p_edge,  &bf, bt, nb, &ex, batch_i); 

        // --- Scan the outputs
        add_content(&addhoc, &bf, bt, nb, &ex, batch_i);
        add_content(&md -> m_p_undef, &bf, bt, nb, &ex, batch_i); 
        for (size_t l(0); l < bf.size(); ++l){
            for (size_t i(0); i < content -> size(); ++i){
                if ((*content)[i].variable_name == "edge_index"){bf[l][i] -= std::get<0>(bf[l][i].min({0}));}
                (*content)[i].process(&bf[l][i], nullptr, t); 
            }
            t -> Fill();
        }
        if (!x){t -> StopCacheLearningPhase();}
        if (!(batch_i.size() + (*prg) >= ds)){(*prg) += batch_i.size(); continue;}
        (*msg) = "\033[1;32m (Done) " + (*msg) + "\033[0m";
    }

    delete md; md = nullptr; 
    t -> Write("", TObject::kOverwrite);
    t -> ResetBranchAddresses(); 
    f -> Close(); 
    f -> Delete(); 
    delete f;
    delete content; content = nullptr; 
    (*prg) = ds;
}

void analysis::execution_metric(metric_t* mt, metric_template* mtx, size_t* prg, std::string* msg){
    metric_template* mx = mtx -> clone(); 
    mx -> _outdir = mtx -> _outdir; 
    mtx -> execute(mt, mx, prg, msg);
    delete mx; 
}
