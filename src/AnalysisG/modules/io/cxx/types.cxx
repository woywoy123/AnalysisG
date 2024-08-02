#include <io.h>

H5::CompType io::member(folds_t){
    H5::CompType pairs(sizeof(folds_t)); 
    pairs.insertMember("k"       , HOFFSET(folds_t, k)       , H5::PredType::NATIVE_INT); 
    pairs.insertMember("index"   , HOFFSET(folds_t, index)   , H5::PredType::NATIVE_INT); 
    pairs.insertMember("is_train", HOFFSET(folds_t, is_train), H5::PredType::NATIVE_HBOOL); 
    pairs.insertMember("is_valid", HOFFSET(folds_t, is_valid), H5::PredType::NATIVE_HBOOL); 
    pairs.insertMember("is_eval" , HOFFSET(folds_t, is_eval) , H5::PredType::NATIVE_HBOOL); 
    return pairs;
}

H5::CompType io::member(graph_hdf5_w inpt){
    H5::CompType pairs(sizeof(graph_hdf5_w)); 

    H5::StrType h5_string(H5::PredType::C_S1, H5T_VARIABLE);
    pairs.insertMember("num_nodes"  , HOFFSET(graph_hdf5_w, num_nodes  ), H5::PredType::NATIVE_INT ); 
    pairs.insertMember("event_index", HOFFSET(graph_hdf5_w, event_index), H5::PredType::NATIVE_LONG); 
    pairs.insertMember("hash"       , HOFFSET(graph_hdf5_w, hash       ), h5_string); 
    pairs.insertMember("filename"   , HOFFSET(graph_hdf5_w, filename   ), h5_string); 

    pairs.insertMember("edge_index"     , HOFFSET(graph_hdf5_w, edge_index     ), h5_string); 
    pairs.insertMember("data_map_graph" , HOFFSET(graph_hdf5_w, data_map_graph ), h5_string); 
    pairs.insertMember("data_map_node"  , HOFFSET(graph_hdf5_w, data_map_node  ), h5_string); 
    pairs.insertMember("data_map_edge"  , HOFFSET(graph_hdf5_w, data_map_edge  ), h5_string); 
    pairs.insertMember("truth_map_graph", HOFFSET(graph_hdf5_w, truth_map_graph), h5_string); 
    pairs.insertMember("truth_map_node" , HOFFSET(graph_hdf5_w, truth_map_node ), h5_string); 
    pairs.insertMember("truth_map_edge" , HOFFSET(graph_hdf5_w, truth_map_edge ), h5_string); 
    pairs.insertMember("data_graph"     , HOFFSET(graph_hdf5_w, data_graph     ), h5_string); 
    pairs.insertMember("data_node"      , HOFFSET(graph_hdf5_w, data_node      ), h5_string); 
    pairs.insertMember("data_edge"      , HOFFSET(graph_hdf5_w, data_edge      ), h5_string); 
    pairs.insertMember("truth_graph"    , HOFFSET(graph_hdf5_w, truth_graph    ), h5_string); 
    pairs.insertMember("truth_node"     , HOFFSET(graph_hdf5_w, truth_node     ), h5_string); 
    pairs.insertMember("truth_edge"     , HOFFSET(graph_hdf5_w, truth_edge     ), h5_string); 
    return pairs;
}
