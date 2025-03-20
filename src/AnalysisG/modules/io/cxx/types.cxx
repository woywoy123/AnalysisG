#include "io.h"

hid_t io::member(folds_t){
    hid_t px = H5Tcreate(H5T_COMPOUND, sizeof(folds_t)); 

    H5Tinsert(px, "k"       , HOFFSET(folds_t, k)       , H5T_NATIVE_INT); 
    H5Tinsert(px, "is_train", HOFFSET(folds_t, is_train), H5T_NATIVE_HBOOL); 
    H5Tinsert(px, "is_valid", HOFFSET(folds_t, is_valid), H5T_NATIVE_HBOOL); 
    H5Tinsert(px, "is_eval" , HOFFSET(folds_t, is_eval) , H5T_NATIVE_HBOOL); 

    hid_t ss = H5Tcopy(H5T_C_S1); 
    H5Tset_size(ss, H5T_VARIABLE); 
    H5Tinsert(px, "hash", HOFFSET(folds_t, hash), ss); 
    return px;
}

hid_t io::member(graph_hdf5_w inpt){

    hid_t px = H5Tcreate(H5T_COMPOUND, sizeof(inpt)); 
    H5Tinsert(px, "num_nodes"   , HOFFSET(graph_hdf5_w, num_nodes   ), H5T_NATIVE_INT ); 
    H5Tinsert(px, "event_index" , HOFFSET(graph_hdf5_w, event_index ), H5T_NATIVE_LONG); 
    H5Tinsert(px, "event_weight", HOFFSET(graph_hdf5_w, event_weight), H5T_NATIVE_DOUBLE); 

    hid_t ss = H5Tcopy(H5T_C_S1); 
    H5Tset_size(ss, H5T_VARIABLE); 
    H5Tinsert(px, "hash"           , HOFFSET(graph_hdf5_w, hash           ), ss); 
    H5Tinsert(px, "filename"       , HOFFSET(graph_hdf5_w, filename       ), ss); 
    H5Tinsert(px, "edge_index"     , HOFFSET(graph_hdf5_w, edge_index     ), ss); 
    H5Tinsert(px, "data_map_graph" , HOFFSET(graph_hdf5_w, data_map_graph ), ss); 
    H5Tinsert(px, "data_map_node"  , HOFFSET(graph_hdf5_w, data_map_node  ), ss); 
    H5Tinsert(px, "data_map_edge"  , HOFFSET(graph_hdf5_w, data_map_edge  ), ss); 
    H5Tinsert(px, "truth_map_graph", HOFFSET(graph_hdf5_w, truth_map_graph), ss); 
    H5Tinsert(px, "truth_map_node" , HOFFSET(graph_hdf5_w, truth_map_node ), ss); 
    H5Tinsert(px, "truth_map_edge" , HOFFSET(graph_hdf5_w, truth_map_edge ), ss); 
    H5Tinsert(px, "data_graph"     , HOFFSET(graph_hdf5_w, data_graph     ), ss); 
    H5Tinsert(px, "data_node"      , HOFFSET(graph_hdf5_w, data_node      ), ss); 
    H5Tinsert(px, "data_edge"      , HOFFSET(graph_hdf5_w, data_edge      ), ss); 
    H5Tinsert(px, "truth_graph"    , HOFFSET(graph_hdf5_w, truth_graph    ), ss); 
    H5Tinsert(px, "truth_node"     , HOFFSET(graph_hdf5_w, truth_node     ), ss); 
    H5Tinsert(px, "truth_edge"     , HOFFSET(graph_hdf5_w, truth_edge     ), ss); 
    return px;
}
