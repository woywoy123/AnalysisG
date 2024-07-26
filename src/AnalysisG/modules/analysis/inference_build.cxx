#include <generators/analysis.h>
#include <tools/vector_cast.h>
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

void execution(model_template* md, std::vector<graph_t*>* data, std::string output, std::vector<variable_t>* content, float* prg){
    TFile* f = TFile::Open(output.c_str(), "RECREATE");
    TTree* t = new TTree("nominal", "data"); 
    std::string msg = "Running model " + std::string(md -> name) + " with sample -> " + output; 
    notification tx = notification(); 

    for (size_t x(0); x < data -> size(); ++x){
        int index = 0; 
        for (size_t i(0); i < content -> size(); ++i){(*content)[i].flush();}
        md -> forward((*data)[x], false); 

        // --- Scan the inputs
        index = add_content(&md -> m_i_graph, content, index, t); 
        index = add_content(&md -> m_i_node, content, index, t); 
        index = add_content(&md -> m_i_edge, content, index, t); 

        // --- Scan the outputs
        index = add_content(&md -> m_p_graph, content, index, t); 
        index = add_content(&md -> m_p_node, content, index, t); 
        index = add_content(&md -> m_p_edge, content, index, t); 

        std::map<std::string, torch::Tensor*> addhoc;
        addhoc["edge_index"] = (*data)[x] -> get_edge_index(md);
        index = add_content(&addhoc, content, index, t);
        index = add_content(&md -> m_p_undef, content, index, t); 

        t -> Fill();

        *prg = float(x)/float(data -> size()); 
    }
    t -> ResetBranchAddresses(); 
    t -> Write();
    delete f; 

    *prg = 1; 
    for (size_t i(0); i < content -> size(); ++i){(*content)[i].purge();}
    delete content; 
}

void analysis::build_inference(){
    this -> success("Starting the model inference.");
    std::map<std::string, std::vector<graph_t*>>* dl = this -> loader -> get_inference(); 
    
    this -> success("Sorted events by event index. Preparing for multithreading.");
    int smpls = dl -> size(); 
    int modls = this -> model_inference.size(); 

    std::map<std::string, std::vector<graph_t*>>::iterator its; 
    std::map<std::string, model_template*>::iterator itm = this -> model_inference.begin(); 

    this -> info("Transferring graphs to device: " + std::string(itm -> second -> device)); 
    torch::TensorOptions* dev = itm -> second -> m_option; 
    this -> loader -> datatransfer(dev, 10); 
    this -> success("Completed transfer");

    std::vector<model_template*> th_models = std::vector<model_template*>(smpls*modls, nullptr); 
    std::vector<std::thread*> th_prc = std::vector<std::thread*>(smpls*modls, nullptr); 
    std::vector<float> th_prg = std::vector<float>(smpls*modls, 0); 

    this -> info("------------- Cloning Models -------------"); 
    for (size_t x(0); x < th_models.size(); ++x){
        int mdx = x%smpls; 
        if (!mdx){its = dl -> begin();}
        if (x && !mdx){++itm;}

        model_settings_t mds; 
        itm -> second -> clone_settings(&mds); 

        this -> success("Cloned model: " + std::string(itm -> second -> name)); 
        model_template* md = itm -> second -> clone(); 
        md -> import_settings(&mds); 

        if(!md -> restore_state()){
            this -> warning("File not found under specified checkpoint path. Skipping"); 
            continue;
        }

        std::string fname = this -> m_settings.output_path + "/" + itm -> first + "/"; 
        meta* mt = this -> tracer -> get_meta_data(its -> first); 
        if (mt){fname += mt -> meta_data.DatasetName + "/";}
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
        addhoc["edge_index"] = its -> second[0] -> get_edge_index(md);
        index = add_content(&addhoc, content, index); 
        index = add_content(&md -> m_p_undef, content, index); 

        th_prc[x] = new std::thread(execution, md, &its -> second, fname, content, &th_prg[x]);
        th_models[x] = md; 
        ++its;
    } 

    std::string msg = "Model Inference Progress"; 
    for (size_t x(0); x < th_prc.size(); ++x){
        while (true){
            bool status = true; 
            float total = 0; 
            for (size_t i(0); i < th_prc.size(); ++i){
                total += th_prg[i]; 
                status *= (th_prg[i] > 0.99); 
            }
            total = total / float(th_prc.size()); 
            this -> progressbar(total, msg); 
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
            if (!status){continue;}
            break; 
        }

        if (th_prc[x] -> joinable()){th_prc[x] -> join();}
        delete th_prc[x]; 
        delete th_models[x]; 
    }
    std::cout << "" << std::endl;
    this -> success("Model inference completed!"); 
}
