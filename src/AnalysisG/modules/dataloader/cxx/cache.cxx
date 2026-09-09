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
        (*tr -> maxlength) = datax.size(); 
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

    auto validate = [this](std::map<std::string, graph_t*>* data, bool* state, tracing_t* thr_){
        bool vld = true; (*thr_ -> maxlength) = data -> size();  
        thr_ -> info("CHECKING HASHES"); 
        std::map<std::string, graph_t*>::iterator itr = data -> begin(); 
        for (; itr != data -> end(); ++itr){
            thr_ -> next(); 
            graph_t* dt = itr -> second; 
            if (!this -> hash_map.count(*dt -> hash)){vld = false; break;}
            long hid = this -> hash_map[*dt -> hash];
            std::string* hxt = (*this -> data_set)[hid] -> hash; 
            vld *= (*dt -> hash) == (*hxt); 
            if (vld){thr_ -> success("[VALID]"); continue;}
            thr_ -> success("[INVALID]: EXPECTED -> " + *hxt + " GOT -> " + *dt -> hash);
            this -> rate_time(1); 
            break;
        } 
        thr_ -> finished(); 
        (*state) = vld; 
        this -> mflush(data);
        this -> pflush(&data);  
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
            this -> pflush(&itr -> second); 
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
    std::map<std::string, graph_t*>* restored = this -> restore_graphs_(pth_verify, threads, true); 
 
    bool valid = restored -> size() > 0;  
    thr = this -> make_threads(1, 1); 
    tracing_t* tr = (*thr -> traces)[0]; 
    tr -> register_thread( new std::thread(validate, restored, &valid, tr), 1); 
    while (this -> await_threads(thr, true)){}
    if (valid){this -> success("Graph cache has been validated!"); return true;}
    this -> failure("The stored cache could not be verified, manually delete the cache folder"); 
    return false;
}

std::map<std::string, graph_t*>* dataloader::restore_graphs_(std::vector<std::string> cache_, int threads, bool force_load){
    auto threaded_reader = [this](
            std::string fname, std::vector<graph_t*>* c_gr, 
            const std::vector<folds_t>* data_k, bool forced_, 
            tracing_t* th_
    ) -> void {
        std::map<std::string, int> load_hash; 

        io ior = io();
        ior.start(fname, "read"); 
        std::vector<std::string> kfold_r; 
        const std::vector<std::string> data_set_ = ior.dataset_names(); 
        fname = this -> get_splits(&fname, "/"); 
        (*th_ -> maxlength) = data_set_.size() + ( (!data_k) ? 0 : data_k -> size() ); 
        if (data_k && !forced_){
            th_ -> info("[Reading][k-fold][" + fname + "]"); 
            const bool eval  = this -> setting -> evaluation; 
            const bool fold  = this -> setting -> validation;
            const bool train = this -> setting -> training; 
            const std::vector<int>* kv = &this -> setting -> kfold; 
            for (size_t x(0); x < data_k -> size(); ++x){
                const folds_t* kf = &(*data_k)[x]; 
                std::string hash = std::string(kf -> hash); 
                th_ -> next();
                if (load_hash[hash] > 0){continue;}
                if (load_hash[hash] == 2 && eval ){continue;}
                if (load_hash[hash] == 3 && train){continue;}
                if (load_hash[hash] == 4 && fold ){continue;}
                int* vl = &load_hash[hash]; 
                (*vl)  = (kf -> is_eval && eval) * 2; 
                if ((*vl) == 2){continue;}
                for (size_t k(0); k < kv -> size(); ++k){
                    if ((*kv)[k] != kf -> k + 1){continue;}
                    *vl = 3 * train * kf -> is_train; 
                    *vl = 4 * fold  * kf -> is_valid; 
                    break;
                }
            }
            th_ -> info("[Finished][k-fold][" + fname + "]");
        }

        if (forced_){(*th_ -> maxlength) = data_set_.size();}
        this -> rate_time(1); 
        th_ -> info("[Mapping][" + fname + "]");
        for (size_t x(0); x < data_set_.size(); ++x){
            std::string hx =  data_set_[x]; 
            if (!load_hash[hx] && !data_k && !forced_){continue;}
            if (this -> hash_map.count(hx) && !forced_){continue;}
            kfold_r.push_back(hx); 
            th_ -> next(); 
        }
        th_ -> info("[Finished][Mapping][" + fname + "]");
        this -> rate_time(1); 

        (*th_ -> idx) = 0;
        (*th_ -> maxlength) = kfold_r.size(); 
        th_ -> success("[Graphs](" + this -> to_string(kfold_r.size()) + ")[" + fname + "]");
        c_gr -> assign(kfold_r.size(), nullptr);  
        for (size_t x(0); x < kfold_r.size(); ++x){
            std::string hx = kfold_r[x]; 
            graph_hdf5_w datar = graph_hdf5_w(); 
            ior.read(&datar, hx);
            graph_hdf5 w = graph_hdf5(); 
            datar.import_gr(&w); 
            datar.flush_data(); 
            graph_t* gx = new graph_t();
            gx -> deserialize(&w); 
            (*c_gr)[x] = gx;
            th_ -> next(); 
            if (x){continue;}
            this -> rate_time(1); 
            th_ -> success("[Reading] " + fname); 
        }
        ior.end();
        th_ -> finished(); 
    }; 



    std::vector<std::string> cache_io = {}; 
    std::map<std::string, std::vector<std::string>> data_set_; 
    for (size_t x(0); x < cache_.size(); ++x){
        std::string fname = cache_[x]; 
        std::string fname_ = this -> get_splits(&fname, "/"); 
        if (!this -> has_string(&fname_, "0x")){continue;}
        if ( this -> has_string(&fname_, ".0x")){continue;}
        cache_io.push_back(fname); 
    }

    std::string path = this -> setting -> training_dataset; 
    std::vector<folds_t> data_k = {}; 
    io io_g = io(); 
    io_g.start(path, "read"); 
    io_g.read(&data_k, "kfolds"); 
    io_g.end();
 
    multithreaded_t* th = this -> make_threads(cache_io.size(), threads); 
    std::vector<std::vector<graph_t*>*> cache_rebuild(cache_io.size(), nullptr); 
    for (size_t x(0); x < cache_io.size(); ++x){
        std::string fname = cache_io[x]; 
        cache_rebuild[x] = new std::vector<graph_t*>(); 
        tracing_t* th_ = th -> traces -> at(x); 
        th_ -> register_thread(new std::thread(threaded_reader, fname, cache_rebuild[x], &data_k, force_load, th_), 10); 
        while (this -> await_threads(th, false)){}
    } 
    while (this -> await_threads(th, true)){}
    for (size_t x(0); x < data_k.size(); ++x){data_k[x].flush_data();}

    std::map<std::string, graph_t*>* restored = new std::map<std::string, graph_t*>(); 
    for (size_t x(0); x < cache_rebuild.size(); ++x){
        std::vector<graph_t*>* datax = cache_rebuild[x]; 
        if (!datax){continue;}
        for (size_t p(0); p < datax -> size(); ++p){
            graph_t* gr = (*datax)[p]; 
            (*restored)[(*gr -> hash)] = gr;
        }
        this -> pflush(&datax); 
    }
    cache_rebuild.clear(); 
    return restored; 
}

void dataloader::restore_graphs(std::vector<std::string> path, int threads, bool force_load){
    std::map<std::string, graph_t*>* restored = this -> restore_graphs_(path, threads, force_load); 
    
    size_t ln = 0; size_t lr = 0;  
    std::map<std::string, graph_t*>::iterator itr; 
    for (itr = restored -> begin(); itr != restored -> end(); ++itr){
        graph_t* gr = itr -> second; 
        if (gr -> preselection && !force_load){lr++; continue;}
        this -> extract_data(gr);
        (*restored)[itr -> first] = nullptr; ++ln; 
    }
    this -> success("Restored " + std::to_string(ln) + " and Rejected " + std::to_string(lr) + " Graphs from cache!"); 
    this -> mflush(restored); 
    this -> pflush(&restored); 
}

void dataloader::restore_graphs(std::string path, int threads, bool force_load){
    bool ish5 = this -> ends_with(&path, ".h5"); 
    std::vector<std::string> files = (ish5) ? std::vector<std::string>({path}) : this -> ls(path, ".h5");
    this -> restore_graphs(files, threads, force_load);
}


