#include <AnalysisG/analysis.h>
#include <ROOT/RDataFrame.hxx>
#include <tools/vector_cast.h>

void analysis::build_inference(){
    int threads_ = this -> m_settings.threads; 
    this -> success("+=============================+"); 
    this -> success("|Starting the model inference.|");
    this -> success("+=============================+"); 
    std::map<std::string, std::vector<graph_t*>>* dl = this -> loader -> get_inference(); 
    
    this -> success("Sorted events by event index. Preparing for multithreading.");
    size_t dev_i = 0; 
    size_t smpls = dl -> size(); 
    size_t modls = this -> model_inference.size(); 

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

    std::map<int, torch::TensorOptions*> ops; 
    std::map<std::string, model_template*>::iterator itm = this -> model_inference.begin(); 
    for (; itm != this -> model_inference.end(); ++itm){
        torch::TensorOptions* opx = itm -> second -> m_option; 
        int dx = itm -> second -> device_index; 
        if (ops.count(dx)){continue;}
        ops[dx] = opx; 
    }
    this -> loader -> datatransfer(&ops); 

    gErrorIgnoreLevel = 6001;
    ROOT::EnableImplicitMT(threads_); 

    int para = 0; 
    its = dl -> begin(); 
    std::thread* thr_ = nullptr;
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
        std::vector<std::string> fnames = this -> split(its -> first, "/");
        (*mdl_title[x]) = fnames[fnames.size()-1] + " | " + std::string(md -> name); 

        fname += fnames[fnames.size()-2] + "/";
        this -> create_path(fname);
        fname += fnames[fnames.size()-1]; 

        // perform a dummy inference with the GNN on a single graph data point
        md -> forward(its -> second[0], false); 
          
        // ------ Get the input maximal size --------- //
        size_t sx = 0; 
        sx += md -> m_i_graph.size() + md -> m_p_graph.size(); 
        sx += md -> m_i_node.size()  + md -> m_p_node.size();
        sx += md -> m_i_edge.size()  + md -> m_p_edge.size(); 
        sx += md -> m_p_undef.size() + 2; 

        // -------- fetch the input and output features ------- //
        std::vector<variable_t>* content = new std::vector<variable_t>(); 
        content -> reserve(sx);

        std::map<std::string, meta*>::iterator itt = this -> meta_data.begin();
        for (; itt != this -> meta_data.end(); ++itt){
            std::cout << fname << " | " << itt -> first << std::endl; 
            abort(); 
        }

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
                if (!th_prc[t] && batched_data[t]){this -> loader -> safe_delete(batched_data[t]);}
                batched_data[t] = nullptr; 
            }
        } 
    } 

    monitor(&th_prc); 
    for (size_t t(0); t < th_prc.size(); ++t){
        if (batched_data[t]){this -> loader -> safe_delete(batched_data[t]);}
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
