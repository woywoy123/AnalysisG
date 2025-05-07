/**
 * @file io.cxx
 * @brief Implementation of the io class methods.
 *
 * This file contains methods for managing input/output operations and importing settings.
 */

#include <io/io.h>
#include <io/cfg.h>
#include <TSystem.h>
#include <thread>

/**
 * @brief Constructor for the io class.
 * 
 * Initializes the io object with the prefix "io".
 */
io::io(){this -> prefix = "io";}

/**
 * @brief Destructor for the io class.
 * 
 * Cleans up resources by closing open files and freeing memory.
 */
io::~io(){
    this -> end();
    this -> root_end(); 
    std::map<std::string, TFile*>::iterator itr = this -> files_open.begin(); 
    for (; itr != this -> files_open.end(); ++itr){
        if (itr -> second -> IsOpen()){itr -> second -> Close();}
        itr -> second -> Delete(); 
        delete itr -> second;
    }

    std::map<std::string, meta*>::iterator itm = this -> meta_data.begin(); 
    for (; itm != this -> meta_data.end(); ++itm){
        if (!itm -> second){continue;}
        delete itm -> second;
    }
    this -> meta_data.clear(); 
}

/**
 * @brief Imports settings into the IO module.
 *
 * This function configures the IO module based on the provided settings.
 * It sets up the metacache path, sum of weights name, and other parameters.
 *
 * @param params Pointer to the settings structure.
 */
void io::import_settings(settings_t* params){
    this -> enable_pyami = params -> fetch_meta; 
    this -> metacache_path = params -> metacache_path; 
    this -> sow_name = params -> sow_name; 
    if (!this -> sow_name.size()){return;}
    this -> info("Checking for Sum of Weights under tree name: " + this -> sow_name); 
}
