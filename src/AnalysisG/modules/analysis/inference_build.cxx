#include <generators/analysis.h>
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
        std::map<std::string, torch::Tensor*>* data, 
        std::vector<std::vector<torch::Tensor>>* buff, 
        torch::Tensor* edge, torch::Tensor* node, torch::Tensor* batch, 
        std::vector<long> mask
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
        for (size_t x(0); x < mask.size(); ++x){
            (*buff)[x].push_back(tn -> index({(*idx) == mask[x]}).clone());
        } 
    }
}

void execution(
        model_template* md, model_settings_t mds, 
        std::vector<graph_t*>* data, size_t* prg,
        std::string output, std::vector<variable_t>* content
){
 
    md = md -> clone(); 
    md -> import_settings(&mds); 
    int ds = 0; 
    for (size_t x(0); x < data -> size(); ++x){ds += (*data)[x] -> batched_events.size();}

    if(!md -> restore_state()){
        md -> failure("Failed to load model: " + md -> model_checkpoint_path); 
        delete content; content = nullptr; 
        delete md; md = nullptr; 
        (*prg) = ds; 
        return; 
    }

    ROOT::EnableImplicitMT(); 
    gErrorIgnoreLevel = kWarning;
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
            delete md; md = nullptr; 
            delete tmp; tmp = nullptr; 
            *prg = ds; 
            return;
        }
        delete tmp; 
    }
 
    TFile* f = TFile::Open(output.c_str(), "RECREATE"); 
    f -> TestBit(TFile::kRecovered);  
    TTree* t = new TTree(mds.tree_name.c_str(), "data");

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
        (*prg) += batch_i.size(); 
        
        for (size_t l(0); l < bf.size(); ++l){
            for (size_t i(0); i < content -> size(); ++i){
                if ((*content)[i].variable_name == "edge_index"){
                    bf[l][i] -= std::get<0>(bf[l][i].min({0}));
                }
                (*content)[i].flush();
                (*content)[i].process(&bf[l][i], nullptr, t); 
            }
            t -> Fill();
        }
    }

    t -> ResetBranchAddresses(); 
    t -> Write("", TObject::kOverwrite);
    f -> Close(); 
    delete f; f = nullptr; 
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
    auto lamb = [](dataloader* ld, torch::TensorOptions* op, int th_){
        ld -> datatransfer(op, th_);
    };

    int threads_ = this -> m_settings.threads; 
    this -> success("+=============================+"); 
    this -> success("|Starting the model inference.|");
    this -> success("+=============================+"); 
    std::map<std::string, std::vector<graph_t*>>* dl = this -> loader -> get_inference(); 
    
    this -> success("Sorted events by event index. Preparing for multithreading.");
    int smpls = dl -> size(); 
    int modls = this -> model_inference.size(); 

    size_t len = 0; 
    std::map<std::string, std::vector<graph_t*>>::iterator its = dl -> begin(); 
    for (; its != dl -> end(); ++its){len += its -> second.size();}
    len *= modls; 

    std::vector<std::thread*> th_prc = std::vector<std::thread*>(smpls*modls, nullptr); 
    std::vector<std::vector<graph_t*>*> batched_data(smpls*modls, nullptr); 
    std::vector<size_t> th_prg(smpls*modls, 0); 
    std::thread* thr_ = nullptr; 

    std::map<std::string, bool> mute; 
    std::map<std::string, bool> device_tr; 
    std::map<std::string, model_template*>::iterator itm; 

    std::map<std::string, std::thread*> trans; 
    itm = this -> model_inference.begin(); 
    for (; itm != this -> model_inference.end(); ++itm){
        std::string dev_n = itm -> second -> device; 
        if (device_tr[dev_n]){continue;}
        this -> info("Transferring graphs to device: " + std::string(dev_n)); 
        device_tr[dev_n] = true;
        trans[dev_n] = new std::thread(lamb, this -> loader, itm -> second -> m_option, 1); 
    }
    std::map<std::string, std::thread*>::iterator ix = trans.begin(); 
    for (; ix != trans.end(); ++ix){ix -> second -> join(); delete ix -> second;}
    this -> success("Transfer Complete!"); 
    trans.clear(); 

    int para = 0; 
    its = dl -> begin(); 
    bool batched = this -> m_settings.batch_size > 1;
    for (size_t x(0); x < th_prc.size(); ++x){
        int mdx = x%modls; 
        if (!mdx){itm = this -> model_inference.begin();}

        model_settings_t mds; 
        model_template* md = itm -> second; 
        md -> clone_settings(&mds);
        md -> inference_mode = true; 
        std::string dev_ = itm -> second -> device; 

        if (!mute[itm -> second-> name]){
            this -> success("Starting model: " + std::string(itm -> second -> name)); 
            mute[itm -> second -> name] = true; 
        }

        if (x && !mdx){++its;}

        std::vector<graph_t*>* grx = nullptr; 
        if (batched){
            grx = this -> loader -> build_batch(&its -> second, itm -> second, nullptr);
            for (size_t i(0); i < its -> second.size(); ++i){its -> second[i] -> in_use = 0;}
        }
        else {grx = &its -> second;}
        batched_data[x] = grx; 

        std::string fname = this -> m_settings.output_path + "/" + itm -> first + "/"; 
        std::vector<std::string> fnames = tools().split(its -> first, "/");
        fname += fnames[fnames.size()-2] + "/";
        this -> create_path(fname);
        fname += fnames[fnames.size()-1]; 

        // perform a dummy inference with the GNN on a single graph data point
        md -> forward(its -> second[0], false); 
   
        // -------- fetch the input and output features ------- //
        int index = 0; 
        std::vector<variable_t>* content = new std::vector<variable_t>(); 

        // --- Scan the inputs
        index = add_content(&md -> m_i_graph, content, index, "g_i_"); 
        index = add_content(&md -> m_i_node, content, index,  "n_i_"); 
        index = add_content(&md -> m_i_edge, content, index,  "e_i_"); 

        // --- Scan the outputs
        index = add_content(&md -> m_p_graph, content, index, "g_o_"); 
        index = add_content(&md -> m_p_node, content, index,  "n_o_"); 
        index = add_content(&md -> m_p_edge, content, index,  "e_o_"); 

        // --- add additional content
        std::map<std::string, torch::Tensor*> addhoc; 
        torch::Tensor ex = its -> second[0] -> get_edge_index(md) -> transpose(0, 1);
        addhoc["edge_index"]   = &ex;
        addhoc[mds.weight_name] = its -> second[0] -> get_event_weight(md); 
        index = add_content(&addhoc, content, index, ""); 
        index = add_content(&md -> m_p_undef, content, index, "extra_"); 
        th_prc[x] = new std::thread(execution, md, mds, batched_data[x], &th_prg[x], fname, content);
        ++itm; ++para; 

        if (!thr_){thr_ = new std::thread(this -> progressbar1, &th_prg, len, "Model Inference Progress");}
        while (para >= threads_){
            for (size_t t(0); t < th_prc.size(); ++t){
                if (!th_prc[t]){continue;}
                if (!th_prc[t] -> joinable()){continue;}
                th_prc[t] -> join(); 
                delete th_prc[t]; 
                th_prc[t] = nullptr; 
                if (batched){flush(batched_data[t]);}
                batched_data[t] = nullptr; 
                --para; 
                break; 
            }
        } 
    } 

    std::string msg = "Model Inference Progress"; 
    for (size_t x(0); x < th_prc.size(); ++x){
        if (!th_prc[x]){continue;}
        th_prc[x] -> join(); 
        delete th_prc[x]; 
        th_prc[x] = nullptr; 
        if (batched){flush(batched_data[x]);}
        batched_data[x] = nullptr;
    }

    for (size_t x(0); x < th_prg.size(); ++x){th_prg[x] = len / modls;}
    for (its = dl -> begin(); its != dl -> end(); ++its){
        its -> second.clear(); 
        its -> second.shrink_to_fit(); 
    }

    dl -> clear();
    delete dl; 
    dl = nullptr; 
    
    if (!thr_){return this -> failure("No models were executed...");}
    thr_ -> join(); 
    delete thr_; thr_ = nullptr; 
    std::cout << "" << std::endl;
    this -> success("Model inference completed!"); 
}
