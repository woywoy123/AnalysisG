/**
 * @file sampletracer.cxx
 * @brief Implementation of the sampletracer class for parallel data processing.
 *
 * This file contains the implementation of the sampletracer class, which provides
 * functionality for concurrent processing of samples using multithreading.
 * It handles task distribution, progress monitoring, and resource cleanup.
 */

#include <generators/sampletracer.h>

/**
 * @brief Constructor for the sampletracer class.
 * 
 * Initializes a new sampletracer instance with default settings.
 * Sets up initial state for containers and flags.
 */
sampletracer::sampletracer(){}

/**
 * @brief Destructor for the sampletracer class.
 * 
 * Cleans up resources used by the sampletracer instance.
 */
sampletracer::~sampletracer(){}

/**
 * @brief Sets the execution mode to silent (no progress output).
 * 
 * When called, this method configures the sampletracer to run without
 * displaying progress information during execution.
 */
void sampletracer::silent(){
    this -> shush = true;
}

/**
 * @brief Sets the execution mode to verbose (with progress output).
 * 
 * When called, this method configures the sampletracer to display
 * progress information during execution.
 */
void sampletracer::verbose(){
    this -> shush = false;
}

/**
 * @brief Sets the output path for the processed data.
 * @param path String containing the directory path.
 * 
 * Configures the directory where the output files will be saved.
 */
void sampletracer::set_output_path(std::string path){
    this -> output_path = path;
}

bool sampletracer::add_meta_data(meta* meta_, std::string filename){
    if (this -> root_container.count(filename)){return false;}
    this -> root_container[filename].add_meta_data(meta_, filename); 
    return true; 
}

meta* sampletracer::get_meta_data(std::string filename){
    if (!this -> root_container.count(filename)){return nullptr;}
    return this -> root_container[filename].get_meta_data(); 
}

std::vector<event_template*> sampletracer::get_events(std::string label){
    std::vector<event_template*> out = {};
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){itr -> second.get_events(&out, label);}
    return out; 
}

bool sampletracer::add_event(event_template* ev, std::string label){
    return this -> root_container[ev -> filename].add_event_template(ev, label); 
}

bool sampletracer::add_graph(graph_template* gr, std::string label){
    return this -> root_container[gr -> filename].add_graph_template(gr, label); 
}

bool sampletracer::add_selection(selection_template* sel){
    return this -> root_container[sel -> filename].add_selection_template(sel); 
}

/**
 * @brief Compiles objects concurrently using multiple threads.
 * @param threads Number of concurrent threads to use for processing.
 * 
 * This is the main processing method that distributes tasks across multiple threads,
 * monitors their progress, and ensures proper cleanup of resources upon completion.
 * It uses lambda functions for task execution and progress monitoring.
 */
void sampletracer::compile_objects(int threads){
    // Lambda function for task execution
    auto lamb = [](size_t* l, int threadidx, container* data){data -> compile(l, threadidx);}; 
    
    // Lambda function for cleanup of progress monitoring resources
    auto flush = [](std::vector<std::string*>* inpt){
        for (size_t x(0); x < inpt -> size(); ++x){delete (*inpt)[x];}
        inpt -> clear(); 
    }; 

    // Initialize progress tracking structures
    std::vector<size_t> progres(this -> root_container.size(), 0); 
    std::vector<size_t> handles(this -> root_container.size(), 0); 
    std::vector<std::string*> titles_(this -> root_container.size(), nullptr); 
    std::vector<std::thread*> threads_(this -> root_container.size(), nullptr); 

    // Set up progress titles and configure containers
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (size_t x(0); itr != this -> root_container.end(); ++itr, ++x){
        progres[x] = itr -> second.len();
        itr -> second.output_path = this -> output_path; 
        std::vector<std::string> vec = this -> split(itr -> first, "/"); 
        titles_[x] = new std::string(vec[vec.size()-1]); 
    }

    // Check if there's any work to be done
    if (!this -> tools::sum(&progres)){
        flush(&titles_); 
        return;
    }

    // Initialize progress monitoring thread
    std::thread* thr = nullptr; 
    if (this -> shush){
        thr = new std::thread(this -> progressbar3, &handles, &progres, nullptr);
        flush(&titles_); 
    }
    else {thr = new std::thread(this -> progressbar3, &handles, &progres, &titles_);}

    // Distribute tasks across available threads
    int tidx = 0; 
    int index = 0; 
    itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr, ++index, ++tidx){
        threads_[index] = new std::thread(lamb, &handles[index], index, &itr -> second); 
        while (tidx > threads-1){tidx = this -> running(&threads_, &handles, &progres);}
    }
    
    // Wait for all threads to complete and clean up
    this -> monitor(&threads_); 
    thr -> join(); delete thr; thr = nullptr; 
}

/**
 * @brief Monitors active threads and returns the number of completed threads.
 * @param threads_ Pointer to vector of thread pointers.
 * @param handles Pointer to vector of handle values.
 * @param progres Pointer to vector of progress values.
 * @return Count of completed threads.
 * 
 * This helper method checks which threads have completed their tasks,
 * joins them to reclaim resources, and counts how many slots are available
 * for new tasks.
 */
int sampletracer::running(std::vector<std::thread*>* threads_, std::vector<size_t>* handles, std::vector<size_t>* progres){
    int running = 0;
    for (size_t x(0); x < threads_ -> size(); ++x){
        std::thread* thr = (*threads_)[x]; 
        if (!thr){continue;} 
        if ((*handles)[x] < (*progres)[x]){running += 1;}
        else {thr -> join(); delete thr; (*threads_)[x] = nullptr;}
    }
    return running;
}

/**
 * @brief Waits for all threads to complete and cleans up resources.
 * @param threads_ Pointer to vector of thread pointers.
 * 
 * This helper method ensures that all active threads complete their execution
 * before the program proceeds further.
 */
void sampletracer::monitor(std::vector<std::thread*>* threads_){
    for (size_t x(0); x < threads_ -> size(); ++x){
        std::thread* thr = (*threads_)[x]; 
        if (!thr){continue;}
        thr -> join(); 
        delete thr;
        (*threads_)[x] = nullptr;
    }
}

void sampletracer::populate_dataloader(dataloader* dl){
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){itr -> second.populate_dataloader(dl);}
}

void sampletracer::fill_selections(std::map<std::string, selection_template*>* inpt){
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){itr -> second.fill_selections(inpt);}
}

