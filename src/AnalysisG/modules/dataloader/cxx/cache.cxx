#include <generators/dataloader.h>
#include <structs/folds.h>
#include <io/io.h>

bool dataloader::dump_graphs(std::string path, int threads){
    auto serialize = [](
            std::vector<graph_t*>* quant, 
            std::vector<std::tuple<graph_hdf5_w, graph_hdf5>>* data_c,
            std::map<std::string, std::vector<int>*>* fname_index, 
            tracing_t* tr
    ){
        tools tl = tools(); 
        for (size_t t(0); t < quant -> size(); ++t){
            graph_t* gr = (*quant)[t]; 
            data_c -> push_back({}); 
            graph_hdf5*  data = &std::get<1>((*data_c)[t]); 
            graph_hdf5_w* grw = &std::get<0>((*data_c)[t]); 

            gr -> serialize(data); 
            data -> export_gr(grw);

            std::string fname = data -> filename; 
            fname = tl.get_splits(&fname, "/");
            std::string hash  = tl.hash(fname);  
            tl.replace(&fname, ".root", ".h5"); 

            fname = (*gr -> graph_name) + "/." + hash + "-" + fname; 
            if (!fname_index -> count(fname)){(*fname_index)[fname] = new std::vector<int>();}
            (*fname_index)[fname] -> push_back(t); 
            (*tr -> coms) = "Serializing: " + fname; 
            tr -> next(); 
        }
        tr -> finished(); 
    };

    auto write = [this](
            std::string fname, 
            std::vector< std::tuple<graph_hdf5_w, graph_hdf5>* > datax, 
            tracing_t* tr
    ){
        io* wrt = new io(); 
        wrt -> start(fname, "write"); 
        std::vector<std::string> spl = this -> split(fname, "/"); 
        tr -> message("Writing HDF5 -> " + spl[spl.size()-1]);  
        for (size_t l(0); l < datax.size(); ++l){
            graph_hdf5_w* h5wrt = &std::get<0>(*datax[l]); 
            graph_hdf5*   h5_   = &std::get<1>(*datax[l]); 
            wrt -> write(h5wrt, h5_ -> hash); 
            tr -> next(); 
        }
        wrt -> end(); 
        delete wrt;     
        tr -> finished(); 
    }; 

    if (!this -> data_set -> size()){this -> warning("Nothing to do. Skipping..."); return true;}
    size_t x = (this -> data_set -> size()/threads); 
    if (this -> data_set -> size() < size_t(threads)){ x = this -> data_set -> size(); }

    // do prealloc 
    std::vector<std::vector<graph_t*>> quant = this -> discretize(this -> data_set, x); 
    std::vector<std::vector<std::tuple<graph_hdf5_w, graph_hdf5>>*> serials(quant.size(), nullptr);  
    std::vector<std::map<std::string, std::vector<int>*>> fnames(quant.size()); 
    for (size_t t(0); t < quant.size(); ++t){
        fnames.push_back(std::map<std::string, std::vector<int>*>());
        serials[t] = new std::vector<std::tuple<graph_hdf5_w, graph_hdf5>>(); 
        serials[t] -> reserve(quant[t].size()); 
    }

    multithreaded_t* thr = this -> make_threads(quant.size(), threads); 
    for (size_t t(0); t < quant.size(); ++t){
        tracing_t* tr = (*thr -> traces)[t]; 
        tr -> register_thread(new std::thread(serialize, &quant[t], serials[t], &fnames[t], tr), quant[t].size() );
    } 
    while (this -> await_threads(thr, true)){}; 
    // sort the graphs to be saves according to their original root name and assure the 
    // sample indexing is consistent. 
    size_t idx = 0; 
    std::map<std::string, std::vector< std::tuple<graph_hdf5_w, graph_hdf5>* >> collect = {}; 
    for (size_t t(0); t < quant.size(); ++t){
        std::map<std::string, std::vector<int>*>::iterator itr; 
        for (itr = fnames[t].begin(); itr != fnames[t].end(); ++itr){
            std::string id = itr -> first; 
            id = (this -> ends_with(&path, "/")) ? path + id : path + "/" + id; 
            for (int i : *itr -> second){collect[id].push_back(&(*serials[t])[i]);}
            idx += itr -> second -> size();  
            delete itr -> second;
        }
    }
    std::vector<std::map<std::string, std::vector<int>*>>().swap(fnames); 
    this -> pflush(&thr); 
   
    size_t dx = 0;  
    thr = this -> make_threads(collect.size(), threads);
    std::vector<std::string> pth_verify(collect.size(), ""); 
    std::map<std::string, std::vector< std::tuple<graph_hdf5_w, graph_hdf5>* >>::iterator itf; 
    for (itf = collect.begin(); itf != collect.end(); ++itf, ++dx){
        pth_verify[dx] = itf -> first; 
        tracing_t* tr = (*thr -> traces)[dx]; 
        tr -> register_thread( new std::thread(write, itf -> first, itf -> second, tr), itf -> second.size()); 
        while (this -> await_threads(thr, false)){}
    }
    while (this -> await_threads(thr, true)){}
    this -> vflush(&serials); 
    this -> pflush(&thr); 

    for (x = 0; x < pth_verify.size(); ++x){
        std::string nx = pth_verify[x];  
        this -> replace(&nx, "/.", "/"); 
        this -> rename(pth_verify[x], nx); 
        pth_verify[x] = nx; 
    }

    this -> info("Validating Graph Cache..."); 
    std::map<std::string, graph_t*>* restored = this -> restore_graphs_(pth_verify, threads); 

    bool valid = true;
    for (x = 0; x < this -> data_set -> size(); ++x){
        graph_t* dt = (*this -> data_set)[x]; 
        valid = valid && restored -> count(*dt -> hash);
        if (valid){continue;}
        break;
    }
    this -> mflush(restored); 
    this -> pflush(&restored);  

    if (valid){this -> success("Graph cache has been validated!"); return true;}
    this -> failure("The stored cache could not be verified, manually delete the cache folder"); 
    return false;
}

std::map<std::string, graph_t*>* dataloader::restore_graphs_(std::vector<std::string> cache_, int threads, bool force_load){
    auto threaded_reader = [](
            std::string pth, std::vector<std::string>* gr_ev, std::vector<graph_t*>* c_gr, size_t* prg
    ){
        io* ior = new io();
        ior -> start(pth, "read"); 
        for (size_t p(0); p < gr_ev -> size(); ++p){
            graph_hdf5_w datar = graph_hdf5_w(); 
            ior -> read(&datar, (*gr_ev)[p]);
            graph_hdf5 w = graph_hdf5(); 
            datar.import_gr(&w); 
            datar.flush_data(); 
            graph_t* gx = new graph_t();
            gx -> deserialize(&w); 
            (*c_gr)[p] = gx;
            (*prg) = p+1; 
        }
        delete ior;
    }; 


    std::vector<folds_t> data_k = {}; 
    std::string path = this -> setting -> training_dataset; 
    if (path.size()){
        io* io_g = new io(); 
        io_g -> start(path  , "read"); 
        io_g -> read(&data_k, "kfolds"); 
        io_g -> end(); 
        delete io_g; 
    }

    std::map<std::string, int> load_hash; 
    bool eval  = this -> setting -> evaluation; 
    bool fold  = this -> setting -> validation;
    bool train = this -> setting -> training; 
    std::vector<int> kv = this -> setting -> kfold; 
    for (size_t x(0); x < data_k.size(); ++x){
        std::string hash = std::string(data_k[x].hash);
        data_k[x].flush_data();
        if (this -> hash_map.count(hash)){continue;}
        if (load_hash.count(hash)){continue;}
    
        if (data_k[x].is_eval && eval){
            load_hash[hash] = 1; 
            continue;
        }
        for (size_t k(0); k < kv.size(); ++k){
            if (kv[k] != data_k[x].k+1){continue;}
            load_hash[hash] = fold * data_k[x].is_valid + train * data_k[x].is_train; 
            break;
        }
    }

    size_t len_cache = 0; 
    std::vector<size_t> trgt = {};
    std::vector<size_t> handles = {};
    std::vector<std::string> cache_io = {}; 
    std::map<std::string, std::vector<std::string>> data_set_; 
    for (size_t x(0); x < cache_.size(); ++x){
        std::string fname = cache_[x]; 
        std::vector<std::string> spl = this -> split(fname, "/"); 

        std::string fname_ = spl[spl.size()-1]; 
        if (!this -> has_string(&fname_, "0x")){continue;}
        cache_io.push_back(fname); 

        io ior = io();
        ior.start(fname, "read"); 

        data_set_[fname] = ior.dataset_names();  
        for (size_t l(0); l < data_set_[fname].size() * (force_load); ++l){load_hash[data_set_[fname][l]] = -1;}
        if (load_hash.size()){
            std::vector<std::string>* check = &data_set_[fname]; 
            std::vector<std::string>::iterator itx = check -> begin(); 
            for (; itx != check -> end();){itx = (load_hash[*itx]) ? ++itx : check -> erase(itx);}
        }

        len_cache += data_set_[fname].size();
        trgt.push_back(data_set_[fname].size()); 
        handles.push_back(0); 
        ior.end(); 
        this -> progressbar(float((x+1)) / float(cache_.size()), "Checking HDF5 size: " + fname_); 
    }

    std::map<std::string, graph_t*>* restored = new std::map<std::string, graph_t*>(); 
    if (!len_cache){return restored;}

    std::string title = "Reading HDF5"; 
    std::thread* prg = new std::thread(this -> progressbar2, &handles, &len_cache, &title); 

    int tidx = 0; 
    std::vector<std::thread*> th_(cache_io.size(), nullptr); 
    std::vector<std::vector<graph_t*>*> cache_rebuild(cache_io.size(), nullptr); 
    for (size_t x(0); x < cache_io.size(); ++x, ++tidx){
        std::vector<std::string> lsx = this -> split(cache_io[x], "/"); 
        title = "Reading HDF5 -> " + lsx[lsx.size()-1]; 
        std::vector<std::string>* gr_ev = &data_set_[cache_io[x]];
        if (!gr_ev -> size()){continue;}

        cache_rebuild[x] = new std::vector<graph_t*>(gr_ev -> size(), nullptr); 
        th_[x] = new std::thread(threaded_reader, cache_io[x], gr_ev, cache_rebuild[x], &handles[x]); 
        while (tidx >= threads){tidx = this -> running(&th_, &handles, &trgt);}
    }
    this -> monitor(&th_); 

    for (size_t x(0); x < cache_rebuild.size(); ++x){
        std::vector<graph_t*>* datax = cache_rebuild[x]; 
        if (!datax){continue;}
        for (size_t p(0); p < datax -> size(); ++p){
            graph_t* gr = (*datax)[p];
            bool pre = gr -> preselection; 
            if (pre){continue;}
            if (force_load){gr -> preselection = true;}
            std::string hash = (*gr -> hash); 
            (*restored)[hash] = gr; 
            (*datax)[p] = nullptr; 
        }
        this -> vflush(datax); 
        this -> pflush(&datax); 
    }
    cache_rebuild.clear(); 
    prg -> join(); delete prg; prg = nullptr; 
    return restored; 
}

void dataloader::restore_graphs(std::vector<std::string> path, int threads, bool force_load){
    std::map<std::string, graph_t*>* restored = this -> restore_graphs_(path, threads, force_load); 
    std::map<std::string, graph_t*>::iterator itr; 
    for (itr = restored -> begin(); itr != restored -> end(); ++itr){
        this -> extract_data(itr -> second);
        (*restored)[itr -> first] = nullptr; 
    }
    this -> success("Restored " + std::to_string(restored -> size()) + " Graphs from cache!"); 
    restored -> clear(); 
    delete restored; 
}

void dataloader::restore_graphs(std::string path, int threads, bool force_load){
    bool ish5 = this -> ends_with(&path, ".h5"); 
    std::vector<std::string> files = (ish5) ? std::vector<std::string>({path}) : this -> ls(path, ".h5");
    this -> restore_graphs(files, threads, force_load);
}


