#include <io.h>

bool io::start(std::string filename, std::string read_write){
    hid_t mode; 
    bool f = this -> is_file(filename); 
    if (!f){this -> create_path(filename);}
    if (this -> file){return true;}
    if (read_write == "write" && !f){mode = H5F_ACC_TRUNC;}
    else if (read_write == "write" && f){mode = H5F_ACC_RDWR;}
    else if (read_write == "read" && f){mode = H5F_ACC_RDONLY;}
    else {return false;}

    this -> file = new H5::H5File(filename, mode);
    return true; 
}

void io::end(){
    if (!this -> file){return;}
    this -> file -> close(); 
    delete this -> file; 
    this -> file = nullptr; 

    std::map<std::string, H5::DataSet*>::iterator itr; 
    for (itr = this -> data_w.begin(); itr != this -> data_w.end(); ++itr){delete itr -> second;}
    for (itr = this -> data_r.begin(); itr != this -> data_r.end(); ++itr){delete itr -> second;}
    this -> data_w.clear(); 
    this -> data_r.clear();
}

H5::DataSet* io::dataset(std::string set_name, H5::CompType type, int length){
    if (!this -> file){return nullptr;}
    if (this -> data_w.count(set_name)){return this -> data_w[set_name];}

    hsize_t dim(length); 
    H5::DataSpace space(1, &dim);
    H5::DataSet* f_ = nullptr; 
    H5::Exception::dontPrint(); 
    try {f_ = new H5::DataSet(this -> file -> openDataSet(set_name));}
    catch (H5::FileIException not_found_error){ 
        f_ = new H5::DataSet(this -> file -> createDataSet(set_name, type, space)); 
    }

    this -> data_w[set_name] = f_; 
    return this -> data_w[set_name]; 
}


herr_t io::file_info(hid_t loc_id, const char* name, const H5L_info_t* linfo, void *opdata){
    hid_t data;
    std::vector<std::string>* names = reinterpret_cast<std::vector<std::string>*>(opdata);
    data = H5Dopen2(loc_id, name, H5P_DEFAULT);
    names -> push_back(std::string(name)); 
    H5Dclose(data);
    return 0;
}


std::vector<std::string> io::dataset_names(){
    if (!this -> file){return {};}
    std::vector<std::string> output; 
    herr_t idx = H5Literate(this -> file -> getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, this -> file_info, &output);
    return output; 
}

bool io::has_dataset_name(std::string name){
    std::vector<std::string> sets = this -> dataset_names(); 
    for (size_t x(0); x < sets.size(); ++x){
        if (name != sets[x]){continue;}
        return true; 
    }
    return false; 
}


H5::DataSet* io::dataset_in_group(H5::Group& group, std::string set_name, H5::CompType type, int length){
    if (!this -> file){return nullptr;}
    if (this -> data_w.count(set_name)){return this -> data_w[set_name];}

    hsize_t dim(length); 
    H5::DataSpace space(1, &dim);
    H5::DataSet* f_ = nullptr; 
    H5::Exception::dontPrint(); 
    try {f_ = new H5::DataSet(group.openDataSet(set_name));}
    catch (H5::FileIException not_found_error){ 
        f_ = new H5::DataSet(group.createDataSet(set_name, type, space)); 
    }

    this -> data_w[set_name] = f_; 
    return this -> data_w[set_name]; 
}

H5::DataSet* io::dataset(std::string set_name){
    if (!this -> file){return nullptr;}
    if (this -> data_r.count(set_name)){return this -> data_r[set_name];}
    H5::Exception::dontPrint(); 
    try {this -> data_r[set_name] = new H5::DataSet(this -> file -> openDataSet(set_name));}
    catch (H5::FileIException not_found_error){return nullptr;}
    return this -> data_r[set_name]; 
}