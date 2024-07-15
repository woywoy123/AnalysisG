#ifndef STRUCTS_FOLDS_H
#define STRUCTS_FOLDS_H

struct folds_t {
    int k = -1; 
    bool is_train = false; 
    bool is_valid = false; 
    bool is_eval = false; 
    int index; 
}; 

#endif
