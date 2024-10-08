#include <generators/analysis.h>
#include <tools/vector_cast.h>
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>

int add_content(std::map<std::string, torch::Tensor*>* data, std::vector<variable_t>* content, int index, TTree* tt = nullptr){
    std::map<std::string, torch::Tensor*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr, ++index){
        if (!tt){content -> push_back(variable_t());}
        (*content)[index].process(itr -> second, itr -> first, tt); 
    }
    return index; 
}

void execution(model_template* md, std::vector<graph_t*>* data, std::string output, std::vector<variable_t>* content, size_t* prg){
    TFile* f = TFile::Open(output.c_str(), "NEW");
    if (!f){
        (*prg) = data -> size(); 
        delete content;
        return;  
    }
    if (f -> IsZombie()){
        delete f; 
        f = TFile::Open(output.c_str(), "RECREATE");
    }

    TTree t = TTree("nominal", "data"); 
    std::string msg = "Running model " + std::string(md -> name) + " with sample -> " + output; 
    notification tx = notification(); 

    torch::AutoGradMode grd(false); 
    for (size_t x(0); x < data -> size(); ++x){
        int index = 0; 
        for (size_t i(0); i < content -> size(); ++i){(*content)[i].flush();}

        md -> forward((*data)[x], false); 

        // --- Scan the inputs
        index = add_content(&md -> m_i_graph, content, index, &t); 
        index = add_content(&md -> m_i_node, content, index, &t); 
        index = add_content(&md -> m_i_edge, content, index, &t); 

        // --- Scan the outputs
        index = add_content(&md -> m_p_graph, content, index, &t); 
        index = add_content(&md -> m_p_node, content, index, &t); 
        index = add_content(&md -> m_p_edge, content, index, &t); 

        std::map<std::string, torch::Tensor*> addhoc;
        addhoc["edge_index"]   = (*data)[x] -> get_edge_index(md);
        addhoc["event_weight"] = (*data)[x] -> get_event_weight(md); 

        index = add_content(&addhoc, content, index, &t);
        index = add_content(&md -> m_p_undef, content, index, &t); 

        t.Fill();

        *prg = x+1;  
    }
    t.ResetBranchAddresses(); 
    t.Write();
    f -> Close(); 
    delete f; 
    delete content; 
}

void analysis::build_inference(){
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

    std::vector<model_template*> th_models = std::vector<model_template*>(smpls*modls, nullptr); 
    std::vector<std::thread*> th_prc = std::vector<std::thread*>(smpls*modls, nullptr); 
    std::vector<size_t> th_prg(smpls*modls, 0); 
    std::thread* thr_ = nullptr; 

    this -> info("------------- Cloning Models -------------"); 
    std::map<std::string, model_template*>::iterator itm; 
    std::map<std::string, bool> mute; 
    std::map<std::string, bool> device_tr; 

    int para = 0; 
    its = dl -> begin(); 
    for (size_t x(0); x < th_models.size(); ++x){
        int mdx = x%modls; 
        if (!mdx){itm = this -> model_inference.begin();}
        if (x && !mdx){++its;}

        if (!device_tr[itm -> second -> device]){
            this -> info("Transferring graphs to device: " + std::string(itm -> second -> device)); 
            torch::TensorOptions* dev = itm -> second -> m_option; 
            this -> loader -> datatransfer(dev, threads_); 
            this -> success("Completed transfer");
            device_tr[itm -> second -> device] = true;
        }

        model_settings_t mds; 
        itm -> second -> clone_settings(&mds); 

        if (!mute[itm -> second-> name]){
            this -> success("Cloned model: " + std::string(itm -> second -> name)); 
            mute[itm -> second -> name] = true; 
        }

        model_template* md = itm -> second -> clone(); 
        md -> import_settings(&mds); 

        if(!md -> restore_state()){
            this -> warning("File not found under specified checkpoint path. Skipping"); 
            continue;
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
        std::map<std::string, torch::Tensor*>::iterator itr; 

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

        th_prc[x] = new std::thread(execution, md, &its -> second, fname, content, &th_prg[x]);
        th_models[x] = md; 
        ++itm; ++para; 

        if (!thr_){thr_ = new std::thread(this -> progressbar1, &th_prg, len, "Model Inference Progress");}

        while (para >= threads_){
            for (size_t t(0); t < th_prc.size(); ++t){
                if (!th_prc[t]){continue;}
                if (!th_prc[t] -> joinable()){continue;}
                th_prc[t] -> join(); 
                delete th_prc[t]; 
                delete th_models[t]; 
                th_models[t] = nullptr; 
                th_prc[t] = nullptr; 
                --para; 
            }
        } 
    } 

    std::string msg = "Model Inference Progress"; 
    for (size_t x(0); x < th_prc.size(); ++x){
        if (!th_prc[x]){continue;}
        th_prc[x] -> join(); 
        delete th_prc[x]; 
        delete th_models[x]; 
        th_prc[x] = nullptr; 
        th_models[x] = nullptr; 
    }
    
    for (its = dl -> begin(); its != dl -> end(); ++its){
        its -> second.clear(); 
        its -> second.shrink_to_fit(); 
    }
    dl -> clear();
    delete dl; 

    if (!thr_){return this -> failure("No models were executed...");}
    thr_ -> join(); 
    delete thr_; 
    std::cout << "" << std::endl;
    this -> success("Model inference completed!"); 
}
