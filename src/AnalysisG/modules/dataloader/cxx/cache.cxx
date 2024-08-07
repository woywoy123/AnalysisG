#include "dataloader.h"
#include <structs/folds.h>
#include <io/io.h>

void dataloader::dump_graphs(std::string path, int threads){
    auto serialize = [](
            std::vector<graph_t*>* quant, 
            std::vector<int>*  quant_idx,
            std::vector<std::tuple<graph_hdf5_w, graph_hdf5>>* data_c,
            std::map<std::string, std::vector<int>*>* fname_index, 
            size_t* handle
    ){
        for (size_t t(0); t < quant -> size(); ++t){
            graph_t* gr = (*quant)[t]; 
            gr -> event_index = (*quant_idx)[t]; 
            data_c -> push_back({}); 
            graph_hdf5* data = &std::get<1>((*data_c)[t]); 
            gr -> serialize(data); 

            graph_hdf5_w* grw      = &std::get<0>((*data_c)[t]); 
            grw -> num_nodes       = data -> num_nodes; 
            grw -> event_index     = data -> event_index;
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
            if (!fname_index -> count(fname)){(*fname_index)[fname] = new std::vector<int>();}
            (*fname_index)[fname] -> push_back(t); 
            *handle = t+1; 
        }
    };


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
            std::string id = this -> hash(itr -> first) + "-" + spl[spl.size()-1]; 
            this -> replace(&id, ".root", ".h5"); 
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
    std::map<std::string, std::vector< std::tuple<graph_hdf5_w, graph_hdf5>* >>::iterator itf; 
    for (itf = collect.begin(); itf != collect.end(); ++itf, ++dx){
        
        io* wrt = new io(); 
        wrt -> start(itf -> first, "write"); 

        title = "Writing HDF5 -> " + itf -> first; 
        std::vector< std::tuple<graph_hdf5_w, graph_hdf5>* > datax = itf -> second; 
        for (size_t l(0); l < datax.size(); ++l){
            graph_hdf5_w* h5wrt = &std::get<0>(*datax[l]); 
            graph_hdf5*   h5_   = &std::get<1>(*datax[l]); 
            wrt -> write(h5wrt, h5_ -> hash); 
            handles[dx] = l+1; 
        }
        wrt -> end(); 
        delete wrt; 
    }
    for (size_t t(0); t < quant.size(); ++t){delete serials[t];}
    prg -> join(); delete prg; 
}

bool dataloader::restore_graphs(std::string path, int threads){
    auto deserialize = [](std::vector<graph_t*>* quant, std::vector<graph_hdf5>* data_r){
        for (size_t t(0); t < data_r -> size(); ++t){
            graph_t* gx = new graph_t();
            gx -> deserialize(&(*data_r)[t]); 
            (*data_r)[t] = graph_hdf5(); 
            (*quant)[t] = gx;
        }
    };

    size_t len_cache = 0; 
    std::vector<size_t> handles = {};

    std::map<std::string, std::vector<std::string>> data_set; 

    std::vector<std::string> cache_io = {}; 
    std::vector<std::string> cache_ = this -> ls(path, ".h5");

    for (size_t x(0); x < cache_.size(); ++x){
        std::string fname = cache_[x]; 
        std::vector<std::string> spl = this -> split(fname, "/"); 

        std::string fname_ = spl[spl.size()-1]; 
        if (!this -> has_string(&fname_, "0x")){continue;}
        cache_io.push_back(fname); 

        io* ior = new io();
        ior -> start(fname, "read"); 
        data_set[fname] = ior -> dataset_names(); 
        len_cache += data_set[fname].size();
        handles.push_back(0); 
        ior -> end(); 
        delete ior; 
    }
    
    if (!len_cache){return false;}

    std::string title = "Reading HDF5"; 
    std::thread* prg = new std::thread(this -> progressbar2, &handles, &len_cache, &title); 

    std::vector<std::thread*> th_(cache_io.size(), nullptr); 
    std::vector<std::vector<graph_hdf5>*> data(cache_io.size(), nullptr); 
    std::vector<std::vector<graph_t*>*> cache_rebuild(cache_io.size(), nullptr); 

    for (size_t x(0); x < cache_io.size(); ++x){
        std::vector<std::string> lsx = this -> split(cache_io[x], "/"); 
        title = "Reading HDF5 -> " + lsx[lsx.size()-1]; 
        std::vector<std::string>* gr_ev = &data_set[cache_io[x]]; 
        io* ior = new io();
        ior -> start(cache_io[x], "read");  

        std::vector<graph_hdf5>* gr_c = new std::vector<graph_hdf5>(gr_ev -> size(), graph_hdf5()); 
        std::vector<graph_t*>*   c_gr = new std::vector<graph_t*>(gr_ev -> size(), nullptr); 
        for (size_t p(0); p < gr_ev -> size(); ++p){
            graph_hdf5_w datar; 
            ior -> read(&datar, (*gr_ev)[p]); 

            graph_hdf5* w        = &(*gr_c)[p]; 
            w -> num_nodes       = datar.num_nodes; 
            w -> event_index     = datar.event_index;
            w -> hash            = std::string(datar.hash);          
            w -> filename        = std::string(datar.filename);      
            w -> edge_index      = std::string(datar.edge_index);    
            
            w -> data_map_graph  = std::string(datar.data_map_graph); 
            w -> data_map_node   = std::string(datar.data_map_node); 
            w -> data_map_edge   = std::string(datar.data_map_edge);  

            w -> truth_map_graph = std::string(datar.truth_map_graph);
            w -> truth_map_node  = std::string(datar.truth_map_node);    
            w -> truth_map_edge  = std::string(datar.truth_map_edge);        

            w -> data_graph      = std::string(datar.data_graph);    
            w -> data_node       = std::string(datar.data_node);     
            w -> data_edge       = std::string(datar.data_edge);     

            w -> truth_graph     = std::string(datar.truth_graph);   
            w -> truth_node      = std::string(datar.truth_node);    
            w -> truth_edge      = std::string(datar.truth_edge); 
            handles[x] = p+1; 
        }

        th_[x] = new std::thread(deserialize, c_gr, gr_c); 
        cache_rebuild[x] = c_gr; 
        data[x] = gr_c; 
        ior -> end();
        delete ior; 
    }

    std::vector<int> max_index = {};
    max_index.reserve(len_cache);  
    std::vector<graph_t*> restored(len_cache, nullptr); 
    for (size_t x(0); x < cache_rebuild.size(); ++x){
        if (th_[x]){th_[x] -> join(); delete th_[x];  th_[x] = nullptr;}
        delete data[x]; data[x] = nullptr; 
        std::vector<graph_t*>* datax = cache_rebuild[x]; 
        for (size_t p(0); p < datax -> size(); ++p){
            graph_t* gr = (*datax)[p]; 
            max_index.push_back(gr -> event_index);  
        }
    }
    prg -> join(); delete prg; prg = nullptr; 

    int mx = this -> max(&max_index); 
    if (mx > len_cache){restored = std::vector<graph_t*>(mx+1, nullptr);}

    for (size_t x(0); x < cache_rebuild.size(); ++x){
        std::vector<graph_t*>* datax = cache_rebuild[x]; 
        for (size_t p(0); p < datax -> size(); ++p){
            graph_t* gr = (*datax)[p]; 
            restored[gr -> event_index] = gr; 
        }
        delete datax; 
        cache_rebuild[x] = nullptr; 
    }

    // validate the restored graphs 
    bool valid = true;
    if (this -> data_set -> size()){
        valid = this -> data_set -> size() == restored.size(); 
        for (size_t x(0); x < restored.size(); ++x){
            graph_t* gr_r = restored[x]; 
            if (!gr_r){valid = false; break;}
            valid *= (*gr_r -> hash) == (*(*this -> data_set)[x] -> hash); 
            if (valid){continue;}
            break; 
        } 
        for (size_t x(0); x < restored.size(); ++x){
            if (!restored[x]){continue;}
            restored[x] -> _purge_all();
            delete restored[x]; 
        }
        if (valid){return true;}
        this -> delete_path(path); 
        return false;
    }

    int s = 0; 
    for (size_t x(0); x < restored.size(); ++x){
        if (!restored[x]){continue;}
        this -> extract_data(restored[x]);
        s++; 
    }

    this -> success("Restored " + std::to_string(s) + " Graphs from cache!"); 
    return valid; 
}

