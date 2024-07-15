#include <io.h>

void io::write(std::map<std::string, particle_t>* inpt, std::string set_name){
    this -> write(inpt, set_name); 
}

H5::CompType io::member(particle_t t){
    H5::CompType pairs(sizeof(particle_t)); 
    H5::StrType h5_string(H5::PredType::C_S1, H5T_VARIABLE);

    pairs.insertMember("e"      , HOFFSET(particle_t, e     ), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("mass"   , HOFFSET(particle_t, mass  ), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("px"     , HOFFSET(particle_t, px    ), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("py"     , HOFFSET(particle_t, py    ), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("pz"     , HOFFSET(particle_t, pz    ), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("pt"     , HOFFSET(particle_t, pt    ), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("eta"    , HOFFSET(particle_t, eta   ), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("phi"    , HOFFSET(particle_t, phi   ), H5::PredType::NATIVE_DOUBLE); 
    pairs.insertMember("charge" , HOFFSET(particle_t, charge), H5::PredType::NATIVE_DOUBLE); 

    pairs.insertMember("cartesian", HOFFSET(particle_t, cartesian ), H5::PredType::NATIVE_HBOOL); 
    pairs.insertMember("polar"    , HOFFSET(particle_t, polar     ), H5::PredType::NATIVE_HBOOL); 

    pairs.insertMember("pdgid" , HOFFSET(particle_t, pdgid), H5::PredType::NATIVE_INT); 
    pairs.insertMember("index" , HOFFSET(particle_t, index), H5::PredType::NATIVE_INT); 

    pairs.insertMember("hash"  , HOFFSET(particle_t, hash  ), h5_string); 
    pairs.insertMember("type"  , HOFFSET(particle_t, type  ), h5_string); 
    pairs.insertMember("symbol", HOFFSET(particle_t, symbol), h5_string); 

    return pairs;
}

H5::CompType io::member(folds_t){
    H5::CompType pairs(sizeof(folds_t)); 
    H5::StrType h5_string(H5::PredType::C_S1, H5T_VARIABLE);
    pairs.insertMember("k"       , HOFFSET(folds_t, k), H5::PredType::NATIVE_INT); 
    pairs.insertMember("index"   , HOFFSET(folds_t, index), H5::PredType::NATIVE_INT); 
    pairs.insertMember("is_train", HOFFSET(folds_t, is_train), H5::PredType::NATIVE_HBOOL); 
    pairs.insertMember("is_valid", HOFFSET(folds_t, is_valid), H5::PredType::NATIVE_HBOOL); 
    pairs.insertMember("is_eval" , HOFFSET(folds_t, is_eval), H5::PredType::NATIVE_HBOOL); 
    return pairs;
}
