#include <AnalysisG/analysis.h>
#include <ROOT/RDataFrame.hxx>
#include <tools/vector_cast.h>
#include <TFile.h>
#include <TTree.h>

int add_content(
        std::map<std::string, torch::Tensor*>* data, 
        std::vector<variable_t>* content, int index, 
        std::string prefx, TTree* tt = nullptr
){
    std::map<std::string, torch::Tensor*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr, ++index){
        if (!tt){content -> push_back(variable_t());}
        std::string name = prefx + itr -> first; 
        (*content)[index].process(itr -> second, &name, tt); 
    }
    return index; 
}

void add_content(
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
        if (tn -> size({0}) == ei){idx = &edge_i;}
        else if (tn -> size({0}) == ni){idx = &node_i;}
        else if (tn -> size({0}) == bi){idx = &batch_i;}
        else {continue;}
        for (size_t x(0); x < mask.size(); ++x){(*buff)[x].push_back(tn -> index({(*idx) == mask[x]}).clone());} 
    }
}

void execution(
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
    for (size_t x(0); x < data -> size(); ++x){

        graph_t* gr = (*data)[x]; 
        md -> forward(gr, false);  

        std::vector<std::vector<torch::Tensor>> bf = {};
        std::vector<long> batch_i = gr -> batched_events; 
        if (bf.size() != batch_i.size()){bf.assign(batch_i.size(), {});} 
        torch::Tensor* bt = gr -> get_batched_events(md); 
        torch::Tensor* nb = gr -> get_batch_index(md); 
        torch::Tensor ei  = gr -> get_edge_index(md) -> transpose(0, 1).clone(); 
        torch::Tensor ex  = nb -> index({ei.index({torch::indexing::Slice(), 0})}).clone();

        std::map<std::string, torch::Tensor*> addhoc;
        addhoc["edge_index"]   = &ei; 
        addhoc[mds.weight_name] = (*data)[x] -> get_event_weight(md); 

        if (!x){
            int index = 0; 
            // --- Scan the inputs
            index = add_content(&md -> m_i_graph, content, index, "g_i_", t); 
            index = add_content(&md -> m_i_node, content, index,  "n_i_", t); 
            index = add_content(&md -> m_i_edge, content, index,  "e_i_", t); 

            // --- Scan the outputs
            index = add_content(&md -> m_p_graph, content, index, "g_o_", t); 
            index = add_content(&md -> m_p_node, content, index,  "n_o_", t); 
            index = add_content(&md -> m_p_edge, content, index,  "e_o_", t); 

            // --- add additional content
            index = add_content(&addhoc, content, index, "", t); 
            index = add_content(&md -> m_p_undef, content, index, "extra_", t); 
            t -> StopCacheLearningPhase(); 
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
                (*content)[i].flush();
                (*content)[i].process(&bf[l][i], nullptr, t); 
            }
            t -> Fill();
        }
        if (!(batch_i.size() + (*prg) >= ds)){(*prg) += batch_i.size(); continue;}
        (*msg) = "\033[1;32m (Done) " + (*msg) + "\033[0m";
    }

    t -> ResetBranchAddresses(); 
    t -> Write("", TObject::kOverwrite);
    f -> Close(); 
    f -> Delete(); 
    delete f;
    delete content; content = nullptr; 
    delete md; md = nullptr; 
    (*prg) = ds;
}

void analysis::build_inference(){
    auto flush = [](std::vector<graph_t*>* grx) -> std::vector<graph_t*>* {
        for (size_t x(0); x < grx -> size(); ++x){
            (*grx)[x] -> _purge_all(); 
            delete (*grx)[x]; 
            (*grx)[x] = nullptr;
        }
        grx -> clear();
        grx -> shrink_to_fit(); 
        delete grx; 
        return nullptr; 
    }; 

    auto lamb = [](dataloader* ld, torch::TensorOptions* op, size_t* num_ev, size_t* prg_ev){
        ld -> datatransfer(op, num_ev, prg_ev);
    };

    int threads_ = this -> m_settings.threads; 
    this -> success("+=============================+"); 
    this -> success("|Starting the model inference.|");
    this -> success("+=============================+"); 
    std::map<std::string, std::vector<graph_t*>>* dl = this -> loader -> get_inference(); 
    
    this -> success("Sorted events by event index. Preparing for multithreading.");
    size_t dev_i = 0; 
    size_t smpls = dl -> size(); 
    size_t modls = this -> model_inference.size(); 

    gErrorIgnoreLevel = 6001;
    ROOT::EnableImplicitMT(threads_); 
    std::map<std::string, bool> device_tr; 
    std::vector<size_t> th_prg(smpls*modls, 0); 
    std::vector<size_t> num_data(modls * smpls, 0);
    std::vector<std::thread*> th_prc(smpls*modls, nullptr); 
    std::vector<std::string*> mdl_title(smpls*modls, nullptr); 
    std::vector<std::vector<graph_t*>*> batched_data(smpls*modls, nullptr); 

    std::map<std::string, std::vector<graph_t*>>::iterator its = dl -> begin(); 
    for (; its != dl -> end(); ++its){
        for (size_t x(0); x < modls; ++x, ++dev_i){
            num_data[dev_i] = its -> second.size();
            mdl_title[dev_i] = new std::string(""); 
        }
    }

    std::map<std::string, model_template*>::iterator itm = this -> model_inference.begin(); 
    for (; itm != this -> model_inference.end(); ++itm){device_tr[itm -> second -> device] = false;}
    this -> info("Transferring graphs to device" + std::string((device_tr.size() > 1) ? "s" : "")); 
    std::vector<std::thread*> trans(device_tr.size() , nullptr); 
    std::vector<std::string*> titles(device_tr.size(), nullptr); 
    std::vector<size_t> handles(device_tr.size(), 0);
    std::vector<size_t> num_evn(device_tr.size(), 0); 

    dev_i = 0; 
    for (itm = this -> model_inference.begin(); itm != this -> model_inference.end(); ++itm){
        std::string dev_n = itm -> second -> device; 
        if (device_tr[dev_n]){continue;}
        device_tr[dev_n] = true;
        titles[dev_i] = new std::string("Progress Device: " + std::string(dev_n)); 
        trans[dev_i]  = new std::thread(lamb, this -> loader, itm -> second -> m_option, &num_evn[dev_i], &handles[dev_i]); 
        ++dev_i; 
    }

    std::thread* thr_ = new std::thread(this -> progressbar3, &handles, &num_evn, &titles); 
    this -> monitor(&trans); 
    this -> success("Transfer Complete!"); 
    thr_ -> join(); delete thr_; thr_ = nullptr; 

    int para = 0; 
    its = dl -> begin(); 
    for (size_t x(0); x < th_prc.size(); ++x, ++para){
        int mdx = x%modls; 
        if (!mdx){itm = this -> model_inference.begin();}
        model_settings_t mds; 
        model_template* md = itm -> second; 
        md -> clone_settings(&mds);
        md -> inference_mode = true; 
        if (x && !mdx){++its;}

        std::vector<graph_t*>* grx = nullptr; 
        if (this -> m_settings.batch_size > 1){
            grx = this -> loader -> build_batch(&its -> second, itm -> second, nullptr);
            for (size_t i(0); i < its -> second.size(); ++i){its -> second[i] -> in_use = 0;}
        }
        else {grx = &its -> second;}
        batched_data[x] = grx; 

        std::string fname = this -> m_settings.output_path + "/" + itm -> first + "/"; 
        std::vector<std::string> fnames = tools().split(its -> first, "/");
        (*mdl_title[x]) = fnames[fnames.size()-1] + " | " + std::string(md -> name); 

        fname += fnames[fnames.size()-2] + "/";
        this -> create_path(fname);
        fname += fnames[fnames.size()-1]; 

        // perform a dummy inference with the GNN on a single graph data point
        md -> forward(its -> second[0], false); 
   
        // -------- fetch the input and output features ------- //
        std::vector<variable_t>* content = new std::vector<variable_t>(); 

        // --- Scan the inputs
        int index = 0; 
        index = add_content(&md -> m_i_graph, content, index, "g_i_"); 
        index = add_content(&md -> m_i_node,  content, index, "n_i_"); 
        index = add_content(&md -> m_i_edge,  content, index, "e_i_"); 

        // --- Scan the outputs
        index = add_content(&md -> m_p_graph, content, index, "g_o_"); 
        index = add_content(&md -> m_p_node,  content, index, "n_o_"); 
        index = add_content(&md -> m_p_edge,  content, index, "e_o_"); 

        // --- add additional content
        std::map<std::string, torch::Tensor*> addhoc; 
        torch::Tensor ex = its -> second[0] -> get_edge_index(md) -> transpose(0, 1);
        addhoc["edge_index"]   = &ex;
        addhoc[mds.weight_name] = its -> second[0] -> get_event_weight(md); 
        index = add_content(&addhoc, content, index, ""); 
        index = add_content(&md -> m_p_undef, content, index, "extra_"); 
        th_prc[x] = new std::thread(execution, md, mds, batched_data[x], &th_prg[x], fname, content, mdl_title[x]);
        ++itm; 

        if (!thr_){thr_ = new std::thread(this -> progressbar3, &th_prg, &num_data, &mdl_title);}
        while (para >= threads_){
            para = this -> running(&th_prc);
            for (size_t t(0); t < th_prc.size(); ++t){
                if ( th_prc[t] && batched_data[t]){continue;}
                if (!th_prc[t] && batched_data[t]){flush(batched_data[t]);}
                batched_data[t] = nullptr; 
            }
        } 
    } 

    monitor(&th_prc); 
    for (size_t t(0); t < th_prc.size(); ++t){
        if (batched_data[t]){flush(batched_data[t]);}
        batched_data[t] = nullptr;
    }

    for (its = dl -> begin(); its != dl -> end(); ++its){
        its -> second.clear(); 
        its -> second.shrink_to_fit(); 
    }

    dl -> clear(); delete dl; dl = nullptr; 
    if (!thr_){return this -> failure("No models were executed...");}
    thr_ -> join(); delete thr_; thr_ = nullptr; 
    this -> success("Model inference completed!"); 
}
