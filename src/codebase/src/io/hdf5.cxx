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

H5::Group io::createGroup(const std::string& group_path) {
    H5::Exception::dontPrint();
    H5::Group group;
    try         {H5::Group group = this -> file -> openGroup(group_path.c_str());}
    catch (...) {H5::Group group = this -> file -> createGroup(group_path.c_str());}
    return group;
}

void io::writeAttribute(const std::string& path, const std::string& attr_name, 
                        const H5::DataType& data_type, void* data)
{
    H5::Exception::dontPrint();
    H5::Group group;
    try {group = H5::Group(this -> file -> openGroup(path));} 
    catch (const H5::FileIException& e) {
        std::cout << "Group does not exist: " << path << std::endl;
    }
    H5::Attribute attribute;
    herr_t status = H5Aexists(group.getId(), attr_name.c_str());
    if (status > 0) {
        attribute = group.openAttribute(attr_name);
        attribute.write(data_type, data);
    } else if (status == 0) {
        H5::DataSpace dataspace = H5::DataSpace(H5S_SCALAR);
        attribute = group.createAttribute(attr_name, data_type, dataspace);
        attribute.write(data_type, data);
    }
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


H5::CompType io::member(coin_t t){
    H5::CompType pairs(sizeof(coin_t)); 
    H5::StrType h5_string(H5::PredType::C_S1, 128);

    pairs.insertMember("symbol"                , HOFFSET(coin_t, symbol)                , h5_string); 
    pairs.insertMember("name"                  , HOFFSET(coin_t, name)                  , h5_string); 
    pairs.insertMember("base_symbol"           , HOFFSET(coin_t, base_symbol)           , h5_string); 

    pairs.insertMember("network_symbol"        , HOFFSET(coin_t, network_symbol)        , h5_string); 
    pairs.insertMember("network_name"          , HOFFSET(coin_t, network_name)          , h5_string); 
    pairs.insertMember("is_busy"               , HOFFSET(coin_t, is_busy)               , H5::PredType::NATIVE_HBOOL); 

    pairs.insertMember("token_contract_address", HOFFSET(coin_t, token_contract_address), h5_string); 

    pairs.insertMember("min_withdraw"          , HOFFSET(coin_t, min_withdraw)          , H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("max_withdraw"          , HOFFSET(coin_t, max_withdraw)          , H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("fee_withdraw"          , HOFFSET(coin_t, fee_withdraw)          , H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("withdraw_enabled"      , HOFFSET(coin_t, withdraw_enabled)      , H5::PredType::NATIVE_HBOOL);

    pairs.insertMember("min_confirmations"     , HOFFSET(coin_t, min_confirmations)     , H5::PredType::NATIVE_INT);
    pairs.insertMember("min_deposit"           , HOFFSET(coin_t, min_deposit)           , H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("deposit_enabled"       , HOFFSET(coin_t, deposit_enabled)       , H5::PredType::NATIVE_HBOOL);

    pairs.insertMember("asset_decimals"        , HOFFSET(coin_t, asset_decimals)        , H5::PredType::NATIVE_INT);
    pairs.insertMember("exchange_decimals"     , HOFFSET(coin_t, exchange_decimals)     , H5::PredType::NATIVE_INT);
    pairs.insertMember("balance"               , HOFFSET(coin_t, balance)               , H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("deposit_address"       , HOFFSET(coin_t, deposit_address)       , h5_string); 
    return pairs;
}

H5::CompType io::member(book_t t){    
    H5::CompType meta(sizeof(book_t)); 
    
    hsize_t book_dims[3] = {2, 1024, 7};
    H5::StrType   h5_string(H5::PredType::C_S1, 64);
    H5::ArrayType h5_book(H5::PredType::NATIVE_DOUBLE, 3, book_dims);
    
    meta.insertMember("book"         , HOFFSET(book_t, book)         , h5_book);
    meta.insertMember("bid_len"      , HOFFSET(book_t, bid_len)      , H5::PredType::NATIVE_UINT);    
    meta.insertMember("ask_len"      , HOFFSET(book_t, ask_len)      , H5::PredType::NATIVE_UINT);  
    meta.insertMember("sequence_norm", HOFFSET(book_t, sequence_norm), H5::PredType::NATIVE_LONG); 
    return meta; 
}


H5::CompType io::member(market_t t){
    H5::CompType pairs(sizeof(market_t)); 
    H5::StrType h5_string(H5::PredType::C_S1, 64);

    pairs.insertMember("market"         , HOFFSET(market_t,      market) , h5_string);
    pairs.insertMember("base_asset"     , HOFFSET(market_t,  base_asset) , h5_string); 
    pairs.insertMember("quote_asset"    , HOFFSET(market_t, quote_asset) , h5_string);

    pairs.insertMember("base_precision" , HOFFSET(market_t,  base_precision), H5::PredType::NATIVE_UINT);
    pairs.insertMember("quote_precision", HOFFSET(market_t, quote_precision), H5::PredType::NATIVE_UINT);

    pairs.insertMember("maker_fee"      , HOFFSET(market_t,    maker_fee), H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("taker_fee"      , HOFFSET(market_t,    taker_fee), H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("taker_lp_fee"   , HOFFSET(market_t, taker_lp_fee), H5::PredType::NATIVE_DOUBLE);

    pairs.insertMember("base_minimum"   , HOFFSET(market_t,  base_minimum), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("quote_minimum"  , HOFFSET(market_t, quote_minimum), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("is_trading"     , HOFFSET(market_t,    is_trading), H5::PredType::NATIVE_HBOOL);
   
    return pairs; 
}
 
H5::CompType io::member(delta_t t){
    H5::CompType pairs(sizeof(delta_t)); 
    H5::StrType h5_string(H5::PredType::C_S1, H5T_VARIABLE);
    
    pairs.insertMember("buy_path"         , HOFFSET(delta_t,   buy_path)       , h5_string);
    pairs.insertMember("sell_path"        , HOFFSET(delta_t,  sell_path)       , h5_string);
    pairs.insertMember("algorithm"        , HOFFSET(delta_t,  algo_name)       , h5_string);
    pairs.insertMember("exchange_1"       , HOFFSET(delta_t, exchange_1)       , h5_string); 
    pairs.insertMember("exchange_2"       , HOFFSET(delta_t, exchange_2)       , h5_string); 

    pairs.insertMember("epoch_time"       , HOFFSET(delta_t, epoch_time)       , H5::PredType::NATIVE_LONG); 
    pairs.insertMember("make_trade"       , HOFFSET(delta_t, make_trade)       , H5::PredType::NATIVE_HBOOL);  

    pairs.insertMember("cprofit"          , HOFFSET(delta_t,    cprofit)       , H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("volume"           , HOFFSET(delta_t,     volume)       , H5::PredType::NATIVE_DOUBLE);
    pairs.insertMember("profit_percentage", HOFFSET(delta_t, profit_percentage), H5::PredType::NATIVE_DOUBLE); 
    return pairs; 
}
