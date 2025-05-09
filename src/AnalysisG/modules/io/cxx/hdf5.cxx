#include "io.h"

void io::read(graph_hdf5_w* out, std::string set_name){
    hid_t pairs = this -> member(graph_hdf5_w());
    H5::DataSet* dataset = this -> dataset(set_name);
    if (!dataset){return;}
    H5::DataSpace space_r = dataset -> getSpace();
    hsize_t dim_r[1];
    space_r.getSimpleExtentDims(dim_r); 
    dataset -> read(out, pairs); 
    space_r.close();
    H5Tclose(pairs); 
}

bool io::start(std::string filename, std::string read_write){
    hid_t mode; 
    bool f = this -> is_file(filename); 
    if (!f){this -> create_path(filename);}
    if (this -> file){return true;}
    if (read_write == "write" && !f){mode = H5F_ACC_TRUNC;}
    else if (read_write == "write" && f){mode = H5F_ACC_RDWR;}
    else if (read_write == "read" && f){mode = H5F_ACC_RDONLY;}
    else {return false;}
    this -> file = new H5::H5File(filename.c_str(), mode);
    return true; 
}

void io::end(){
    if (this -> file){
        this -> file -> close(); 
        delete this -> file; 
        this -> file = nullptr; 
    }

    std::map<std::string, H5::DataSet*>::iterator itr; 
    for (itr = this -> data_w.begin(); itr != this -> data_w.end(); ++itr){itr -> second -> close(); delete itr -> second;}
    for (itr = this -> data_r.begin(); itr != this -> data_r.end(); ++itr){itr -> second -> close(); delete itr -> second;}
    this -> data_w.clear(); 
    this -> data_r.clear();
}

H5::DataSet* io::dataset(std::string set_name, hid_t type, long long unsigned int length){
    if (!this -> file){return nullptr;}
    if (this -> data_w.count(set_name)){return this -> data_w[set_name];}

    hsize_t dim[1] = {length}; 
    H5::DataSpace space(1, dim);
    H5::Exception::dontPrint(); 
    this -> data_w[set_name] = new H5::DataSet(this -> file -> createDataSet(set_name.c_str(), type, space)); 
    return this -> data_w[set_name]; 
}

herr_t io::file_info(hid_t loc_id, const char* name, const H5L_info_t*, void *opdata){
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
    H5Literate(this -> file -> getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, this -> file_info, &output);
    return output; 
}

H5::DataSet* io::dataset(std::string set_name){
    if (!this -> file){return nullptr;}
    if (this -> data_r.count(set_name)){return this -> data_r[set_name];}
    this -> data_r[set_name] = new H5::DataSet(this -> file -> openDataSet(set_name.c_str()));
    return this -> data_r[set_name]; 
}
