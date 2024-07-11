#include <generators/analysis.h>
#include <tools/vector_cast.h>
#include <TFile.h>
#include <TTree.h>

void execution(model_template* md, std::vector<graph_t*>* data, std::string output, settings_t* sett){
    std::vector<std::string> mass_targets = sett -> targets;
    std::map<std::string, torch::Tensor*>::iterator itr; 
    TFile* f = TFile::Open(output.c_str(), "RECREATE");
    TTree* t = new TTree("nominal", "data"); 

    std::vector<std::vector<float>> type;
    TBranch* tb = t -> Branch("topology", &type); 

    for (size_t x(0); x < data -> size(); ++x){
        md -> forward((*data)[x], false); 

        // need to make this better.
        itr = md -> m_p_edge.begin();
        for (; itr != md -> m_p_edge.end(); ++itr){
            std::vector<signed long> s = tensor_size(itr -> second); 
            //if (s.size() == 2){
            //std::vector<std::vector<float>> type;
            torch::Tensor ten = itr -> second -> softmax({-1}); 
            tensor_to_vector(&ten, &type, &s, float(0));  
        }
        t -> Fill();

        type.clear(); 





    }
    t -> ResetBranchAddresses(); 
    t -> Write(); 
    delete f; 
    abort(); 
}




void analysis::build_inference(){
    this -> success("Starting the model inference.");
    std::map<std::string, std::vector<graph_t*>>* dl = this -> loader -> get_inference(); 

    this -> success("Sorted events by event index. Preparing for multithreading.");
    int smpls = dl -> size(); 
    int modls = this -> model_inference.size(); 

    std::map<std::string, std::vector<graph_t*>>::iterator its; 
    std::map<std::string, model_template*>::iterator itm = this -> model_inference.begin(); 

    std::vector<model_template*> th_models = std::vector<model_template*>(smpls*modls, nullptr); 
    std::vector<std::thread*> th_prc = std::vector<std::thread*>(smpls*modls, nullptr); 

    gInterpreter -> GenerateDictionary("vector<vector<float> > ", "vector");

    this -> info("------------- Cloning Models -------------"); 
    for (size_t x(0); x < th_models.size(); ++x){
        int mdx = x%smpls; 
        if (!mdx){its = dl -> begin();}
        if (mdx == smpls-1){++itm;}
        model_settings_t mds; 
        itm -> second -> clone_settings(&mds); 
        model_template* md = itm -> second -> clone(); 
        md -> import_settings(&mds); 
        if(!md -> restore_state()){
            this -> warning("File not found under specified checkpoint path. Skipping"); 
            continue;
        }


        std::string fname = this -> m_settings.output_path + "/" + itm -> first + "/"; 
        this -> create_path(fname);
        fname += tools().split(its -> first, "/").back(); 


        execution(md, &its -> second, fname, &this -> m_settings);
        th_models[x] = md; 
        ++its;
    } 

}
