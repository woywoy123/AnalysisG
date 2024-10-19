#include <generators/analysis.h>
#include <ROOT/RDataFrame.hxx>
#include <tools/vector_cast.h>
#include <TFile.h>
#include <TTree.h>

int add_content(
        std::map<std::string, torch::Tensor*>* data, 
        std::vector<variable_t>* content, 
        int index, TTree* tt = nullptr
){
    std::map<std::string, torch::Tensor*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr, ++index){
        if (!tt){content -> push_back(variable_t());}
        std::string name = itr -> first; 
        (*content)[index].process(itr -> second, &name, tt); 
    }
    return index; 
}

void add_content(std::map<std::string, torch::Tensor*>* data, std::vector<torch::Tensor>* buff){
    std::map<std::string, torch::Tensor*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr){buff -> push_back(itr -> second -> clone());}
}

void execution(
        model_template* md, model_settings_t mds, 
        std::vector<graph_t*>* data, size_t* prg,
        std::string output, std::vector<variable_t>* content
){

    ROOT::EnableImplicitMT(); 
 
    TFile* f = new TFile(output.c_str(), "UPDATE"); 
    if (!f){f = new TFile(output.c_str(), "RECREATE");}
    else if (f -> IsZombie()){delete f; f = new TFile(output.c_str(), "RECREATE");}

    TTree* t = (TTree*)f -> Get("nominal"); 
    if (!t){t = new TTree("nominal", "data");}
    size_t l = t -> GetEntries();

    if (l == data -> size()){
        delete content; content = nullptr; 
        delete f; f = nullptr;  
        *prg = data -> size(); 
        return;
    }
    if (l){f -> Delete("*;*");}
    
    md = md -> clone(); 
    md -> import_settings(&mds); 
    if(!md -> restore_state()){
        delete content; content = nullptr; 
        delete md; md = nullptr; 
        delete t; t = nullptr; 
        (*prg) = data -> size(); 
        return; 
    }

    torch::AutoGradMode grd(false); 
    std::map<size_t, std::vector<torch::Tensor>> buffer; 
    for (size_t x(0); x < data -> size(); ++x){
        md -> forward((*data)[x], false);  

        std::vector<torch::Tensor>* bf = &buffer[x]; 
        add_content(&md -> m_i_graph, bf); 
        add_content(&md -> m_i_node, bf); 
        add_content(&md -> m_i_edge, bf); 

        // --- Scan the outputs
        add_content(&md -> m_p_graph, bf); 
        add_content(&md -> m_p_node, bf); 
        add_content(&md -> m_p_edge, bf); 

        std::map<std::string, torch::Tensor*> addhoc;
        addhoc["edge_index"] = (*data)[x] -> get_edge_index(md);
        addhoc["event_weight"] = (*data)[x] -> get_event_weight(md); 
        add_content(&addhoc, bf);
        add_content(&md -> m_p_undef, bf); 
        (*prg) = x+1; 


        if (x != 0){continue;}
        int index = 0; 
        // --- Scan the inputs
        index = add_content(&md -> m_i_graph, content, index, t); 
        index = add_content(&md -> m_i_node, content, index, t); 
        index = add_content(&md -> m_i_edge, content, index, t); 

        // --- Scan the outputs
        index = add_content(&md -> m_p_graph, content, index, t); 
        index = add_content(&md -> m_p_node, content, index, t); 
        index = add_content(&md -> m_p_edge, content, index, t); 

        // --- add additional content
        index = add_content(&addhoc, content, index, t); 
        index = add_content(&md -> m_p_undef, content, index, t); 
    }    
   
    std::map<size_t, std::vector<torch::Tensor>>::iterator itr; 
    for (itr = buffer.begin(); itr != buffer.end(); ++itr){
        for (size_t i(0); i < content -> size(); ++i){
            (*content)[i].flush();
            (*content)[i].process(&itr -> second[i], nullptr, t); 
        }
        t -> Fill();
    }

    t -> ResetBranchAddresses(); 
    t -> Write("", TObject::kOverwrite);
    f -> Close(); 
    delete f; f = nullptr; 
    delete content; content = nullptr; 
    delete md; md = nullptr; 
    (*prg) = data -> size(); 
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
    std::vector<size_t> th_prg(smpls*modls, 0); 
    std::thread* thr_ = nullptr; 

    this -> info("------------- Cloning Models -------------"); 
    std::map<std::string, bool> mute; 
    std::map<std::string, bool> device_tr; 
    std::map<std::string, model_template*>::iterator itm; 
    std::map<std::string, std::vector<graph_t*>*> batched_data; 

    itm = this -> model_inference.begin(); 
    for (; itm != this -> model_inference.end(); ++itm){
        std::string dev_n = itm -> second -> device; 
        if (device_tr[dev_n]){continue;}
        device_tr[dev_n] = true;

        this -> info("Transferring graphs to device: " + std::string(dev_n)); 
        this -> loader -> datatransfer(itm -> second -> m_option, threads_); 
        this -> success("Completed transfer");
    }


    int para = 0; 
    its = dl -> begin(); 
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

        bool batched = this -> m_settings.batch_size > 1;

        if (x && !mdx){
            ++its;
            //if (batched){flush(batched_data[dev_]);}
            //batched_data[dev_] = nullptr; 
        }

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
        index = add_content(&md -> m_i_graph, content, index); 
        index = add_content(&md -> m_i_node, content, index); 
        index = add_content(&md -> m_i_edge, content, index); 

        // --- Scan the outputs
        index = add_content(&md -> m_p_graph, content, index); 
        index = add_content(&md -> m_p_node, content, index); 
        index = add_content(&md -> m_p_edge, content, index); 

        // --- add additional content
        std::map<std::string, torch::Tensor*> addhoc; 
        addhoc["edge_index"]   = its -> second[0] -> get_edge_index(md);
        addhoc["event_weight"] = its -> second[0] -> get_event_weight(md); 
        index = add_content(&addhoc, content, index); 
        index = add_content(&md -> m_p_undef, content, index); 


//        if (batched && !batched_data[dev_]){
//            batched_data[dev_] = this -> loader -> build_batch(&its -> second, itm -> second, nullptr);
//        }
//        else {batched_data[dev_] = &its -> second;}

        th_prc[x] = new std::thread(execution, md, mds, &its -> second, &th_prg[x], fname, content);
        ++itm; ++para; 

        if (!thr_){thr_ = new std::thread(this -> progressbar1, &th_prg, len, "Model Inference Progress");}

        while (para >= threads_){
            for (size_t t(0); t < th_prc.size(); ++t){
                if (!th_prc[t]){continue;}
                if (!th_prc[t] -> joinable()){continue;}
                th_prc[t] -> join(); 
                delete th_prc[t]; 
                th_prc[t] = nullptr; 
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
