#include "dataloader.h"
#include <structs/folds.h>
#include <io/io.h>

bool dataloader::dump_graphs(std::string path, int threads){
    auto serialize = [](
            std::vector<graph_t*>* quant, 
            std::vector<int>*  quant_idx,
            std::vector<std::tuple<graph_hdf5_w, graph_hdf5>>* data_c,
            std::map<std::string, std::vector<int>*>* fname_index, 
            size_t* handle
    ){
        tools tl = tools(); 
        for (size_t t(0); t < quant -> size(); ++t){
            graph_t* gr = (*quant)[t]; 
            data_c -> push_back({}); 
            graph_hdf5* data = &std::get<1>((*data_c)[t]); 
            gr -> serialize(data); 

            graph_hdf5_w* grw      = &std::get<0>((*data_c)[t]); 
            grw -> num_nodes       = data -> num_nodes; 
            grw -> event_index     = data -> event_index;
            grw -> event_weight    = data -> event_weight; 

            grw -> hash            = const_cast<char*>(data -> hash.data());          
            grw -> filename        = const_cast<char*>(data -> filename.data());      
            grw -> edge_index      = const_cast<char*>(data -> edge_index.data());    

            grw -> data_map_graph  = const_cast<char*>(data -> data_map_graph.data()); 
            grw -> data_map_node   = const_cast<char*>(data -> data_map_node.data()); 
            grw -> data_map_edge   = const_cast<char*>(data -> data_map_edge.data());  

            grw -> truth_map_graph = const_cast<char*>(data -> truth_map_graph.data());
            grw -> truth_map_node  = const_cast<char*>(data -> truth_map_node.data());    
            grw -> truth_map_edge  = const_cast<char*>(data -> truth_map_edge.data());        

            grw -> data_graph      = const_cast<char*>(data -> data_graph.data());    
            grw -> data_node       = const_cast<char*>(data -> data_node.data());     
            grw -> data_edge       = const_cast<char*>(data -> data_edge.data());     

            grw -> truth_graph     = const_cast<char*>(data -> truth_graph.data());   
            grw -> truth_node      = const_cast<char*>(data -> truth_node.data());    
            grw -> truth_edge      = const_cast<char*>(data -> truth_edge.data());   

            std::string fname = data -> filename; 
            std::vector<std::string> spl = tl.split(fname, "/"); 
            fname = spl[spl.size()-1]; 
            std::string hash  = tl.hash(fname);  
            tl.replace(&fname, ".root", ".h5"); 
            fname = (*gr -> graph_name) + "/" + hash + "-" + fname; 
            delete gr -> graph_name; gr -> graph_name = nullptr; 
            if (!fname_index -> count(fname)){(*fname_index)[fname] = new std::vector<int>();}
            (*fname_index)[fname] -> push_back(t); 
            *handle = t+1; 
        }
    };

    if (!this -> data_set -> size()){this -> warning("Nothing to do. Skipping..."); return true;}
    int x = (this -> data_set -> size()/threads); 
    std::vector<std::vector<graph_t*>> quant = this -> discretize(this -> data_set, x); 
    std::vector<std::vector<int>>  quant_idx = this -> discretize(this -> data_index, x);
    std::vector<std::vector<std::tuple<graph_hdf5_w, graph_hdf5>>*> serials(quant.size(), nullptr);  

    // do prealloc 
    std::vector<std::map<std::string, std::vector<int>*>> fnames(quant.size()); 
    for (size_t t(0); t < quant.size(); ++t){
        fnames.push_back(std::map<std::string, std::vector<int>*>());
        serials[t] = new std::vector<std::tuple<graph_hdf5_w, graph_hdf5>>(); 
        serials[t] -> reserve(quant[t].size()); 
    }

    std::vector<size_t> handles(quant.size(), 0); 
    std::vector<std::thread*> th_(quant.size(), nullptr); 
    for (size_t t(0); t < th_.size(); ++t){
        th_[t] = new std::thread(serialize, &quant[t], &quant_idx[t], serials[t], &fnames[t], &handles[t]);
    }

    std::string title = "Graph Serialization"; 
    std::thread* prg = new std::thread(this -> progressbar1, &handles, this -> data_set -> size(), title); 

    // sort the graphs to be saves according to their original root name and assure the 
    // sample indexing is consistent. 
    size_t idx = 0; 
    std::map<std::string, std::vector< std::tuple<graph_hdf5_w, graph_hdf5>* >> collect = {}; 
    for (size_t t(0); t < quant.size(); ++t){
        th_[t] -> join(); delete th_[t]; 
        std::vector<std::tuple<graph_hdf5_w, graph_hdf5>>* c_ = serials[t]; 
        std::map<std::string, std::vector<int>*> c_idx = fnames[t]; 

        std::map<std::string, std::vector<int>*>::iterator itr; 
        for (itr = c_idx.begin(); itr != c_idx.end(); ++itr){
            std::vector<std::string> spl = this -> split(itr -> first, "/"); 
            std::string id = itr -> first; 
            id = (this -> ends_with(&path, "/")) ? path + id : path + "/" + id; 
            for (int i : *itr -> second){collect[id].push_back(&(*c_)[i]);}
            idx += itr -> second -> size();  
            delete itr -> second;
        }
    }
    std::vector<std::map<std::string, std::vector<int>*>>().swap(fnames); 
    prg -> join(); delete prg; 
  
    handles = std::vector<size_t>(collect.size(), 0);
    prg = new std::thread(this -> progressbar2, &handles, &idx, &title); 

    int dx = 0; 
    std::vector<std::string> pth_verify; 
    std::map<std::string, std::vector< std::tuple<graph_hdf5_w, graph_hdf5>* >>::iterator itf; 
    for (itf = collect.begin(); itf != collect.end(); ++itf, ++dx){
        
        io* wrt = new io(); 
        wrt -> start(itf -> first, "write"); 
        std::vector<std::string> spl = this -> split(itf -> first, "/"); 
        title = "Writing HDF5 -> " + spl[spl.size()-1]; 
        std::vector< std::tuple<graph_hdf5_w, graph_hdf5>* > datax = itf -> second; 
        for (size_t l(0); l < datax.size(); ++l){
            graph_hdf5_w* h5wrt = &std::get<0>(*datax[l]); 
            graph_hdf5*   h5_   = &std::get<1>(*datax[l]); 
            wrt -> write(h5wrt, h5_ -> hash); 
            handles[dx] = l+1; 
        }
        wrt -> end(); 
        delete wrt; 
        pth_verify.push_back(itf -> first); 
    }
    for (size_t t(0); t < quant.size(); ++t){delete serials[t];}
    prg -> join(); delete prg; 


    this -> info("Validating Graph Cache..."); 
    std::map<std::string, graph_t*>* restored = this -> restore_graphs_(pth_verify, threads); 

    bool valid = true;
    valid = this -> data_set -> size() == restored -> size(); 
    for (size_t x(0); x < this -> data_set -> size(); ++x){
        graph_t* dt = (*this -> data_set)[x]; 
        valid *= restored -> count(*dt -> hash); 
        if (valid){continue;}
        break; 
    }

    std::map<std::string, graph_t*>::iterator itr; 
    for (itr = restored -> begin(); itr != restored -> end(); ++itr){
        itr -> second -> _purge_all();
        delete itr -> second; 
    }
    delete restored; 

    if (valid){this -> success("Graph cache has been validated!"); return true;}
    this -> failure("The stored cache could not be verified, manually delete the cache folder"); 
    return false;
}

std::map<std::string, graph_t*>* dataloader::restore_graphs_(std::vector<std::string> cache_, int threads){
    auto threaded_reader = [](
            std::string pth, 
            std::vector<std::string>* gr_ev, 
            std::vector<graph_t*>* c_gr, 
            size_t* prg
    ){
        io* ior = new io();
        ior -> start(pth, "read"); 
        for (size_t p(0); p < gr_ev -> size(); ++p){
            graph_hdf5_w datar; 
            ior -> read(&datar, (*gr_ev)[p]); 

            graph_hdf5 w      = graph_hdf5(); 
            w.num_nodes       = datar.num_nodes; 
            w.event_index     = datar.event_index;
            w.event_weight    = datar.event_weight; 

            w.hash            = std::string(datar.hash);          
            w.filename        = std::string(datar.filename);      
            w.edge_index      = std::string(datar.edge_index);    
            
            w.data_map_graph  = std::string(datar.data_map_graph); 
            w.data_map_node   = std::string(datar.data_map_node); 
            w.data_map_edge   = std::string(datar.data_map_edge);  

            w.truth_map_graph = std::string(datar.truth_map_graph);
            w.truth_map_node  = std::string(datar.truth_map_node);    
            w.truth_map_edge  = std::string(datar.truth_map_edge);        

            w.data_graph      = std::string(datar.data_graph);    
            w.data_node       = std::string(datar.data_node);     
            w.data_edge       = std::string(datar.data_edge);     

            w.truth_graph     = std::string(datar.truth_graph);   
            w.truth_node      = std::string(datar.truth_node);    
            w.truth_edge      = std::string(datar.truth_edge); 

            graph_t* gx = new graph_t();
            gx -> deserialize(&w); 
            (*c_gr)[p] = gx;
            (*prg) = p+1; 
        }
        ior -> end();
        delete ior; 
    }; 


    std::string path = this -> setting -> training_dataset; 
    std::map<std::string, bool> ignore_hash; 
    if (!this -> setting -> evaluation && path.size()){
        io* io_g = new io(); 
        std::vector<folds_t> data = {}; 
        io_g -> start(path, "read"); 
        io_g -> read(&data, "kfolds"); 
        io_g -> end(); 
        delete io_g; 
        for (size_t x(0); x < data.size(); ++x){
            if (data[x].is_eval){continue;}
            ignore_hash[std::string(data[x].hash)] = true; 
        }
    }

    size_t len_cache = 0; 
    std::vector<size_t> handles = {};
    std::vector<std::string> cache_io = {}; 
    std::map<std::string, std::vector<std::string>> data_set; 
    for (size_t x(0); x < cache_.size(); ++x){

        std::string fname = cache_[x]; 
        std::vector<std::string> spl = this -> split(fname, "/"); 

        std::string fname_ = spl[spl.size()-1]; 
        if (!this -> has_string(&fname_, "0x")){continue;}
        cache_io.push_back(fname); 

        io* ior = new io();
        ior -> start(fname, "read"); 

        std::vector<std::string> check = ior -> dataset_names();  
        if (ignore_hash.size()){
            for (size_t t(0); t < check.size(); ++t){
                if (ignore_hash[check[t]]){continue;}
                data_set[fname].push_back(check[t]); 
            }
        }
        else {data_set[fname] = check;}

        len_cache += data_set[fname].size();
        handles.push_back(0); 
        ior -> end(); 
        delete ior; 
        this -> progressbar(float((x+1)) / float(cache_.size()), "Checking HDF5 size: " + fname_); 
    }

    std::map<std::string, graph_t*>* restored = new std::map<std::string, graph_t*>(); 
    if (!len_cache){return restored;}

    std::string title = "Reading HDF5"; 
    std::thread* prg = new std::thread(this -> progressbar2, &handles, &len_cache, &title); 

    std::vector<std::thread*> th_(cache_io.size(), nullptr); 
    std::vector<std::vector<graph_hdf5>*> data(cache_io.size(), nullptr); 
    std::vector<std::vector<graph_t*>*> cache_rebuild(cache_io.size(), nullptr); 

    for (size_t x(0); x < cache_io.size(); ++x){
        std::vector<std::string> lsx = this -> split(cache_io[x], "/"); 
        title = "Reading HDF5 -> " + lsx[lsx.size()-1]; 
        std::vector<std::string>* gr_ev = &data_set[cache_io[x]]; 
        if (!gr_ev -> size()){continue;}

        std::vector<graph_t*>*     c_gr = new std::vector<graph_t*>(gr_ev -> size(), nullptr); 
        th_[x] = new std::thread(threaded_reader, cache_io[x], gr_ev, c_gr, &handles[x]); 
        cache_rebuild[x] = c_gr; 
        if (x % threads != threads -1){continue;}
        th_[x] -> join(); delete th_[x]; th_[x] = nullptr; 
    }

    for (size_t x(0); x < cache_rebuild.size(); ++x){
        if (th_[x]){th_[x] -> join(); delete th_[x];  th_[x] = nullptr;}
        std::vector<graph_t*>* datax = cache_rebuild[x]; 
        if (!datax){continue;}
        for (size_t p(0); p < datax -> size(); ++p){
            graph_t* gr = (*datax)[p];
            (*restored)[*gr -> hash] = gr;  
        }
        datax -> shrink_to_fit(); 
        delete datax; 
        cache_rebuild[x] = nullptr; 
    }
    prg -> join(); delete prg; prg = nullptr; 
    return restored; 
}

void dataloader::restore_graphs(std::vector<std::string> path, int threads){
    std::map<std::string, graph_t*>* restored = this -> restore_graphs_(path, threads); 

    std::map<std::string, graph_t*>::iterator itr; 
    for (itr = restored -> begin(); itr != restored -> end(); ++itr){this -> extract_data(itr -> second);}
    this -> success("Restored " + std::to_string(restored -> size()) + " Graphs from cache!"); 
    delete restored; 
}

void dataloader::restore_graphs(std::string path, int threads){
    std::vector<std::string> files = this -> ls(path, ".h5"); 
    this -> restore_graphs(files, threads);
}


