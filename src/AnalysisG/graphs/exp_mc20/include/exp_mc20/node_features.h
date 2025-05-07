/** \file node_features.h
 *  \brief Node feature definitions for exp_mc20.
 *
 *  This file contains functions to extract various features from particle nodes.
 */

#ifndef NODES_FEATURES_EXP_MC20_H
#define NODES_FEATURES_EXP_MC20_H

// --------------------- Node Truth --------------------- //
/** \brief Resolves node information. */
void static res_node(int* o, particle_template* p){
//    std::string type = p -> type;
//    if (type == "mu"){*o = ((muon*)p) -> from_res;}
//    else if (type == "el"){*o = ((electron*)p) -> from_res;}
//    else if (type == "jet"){*o = ((jet*)p) -> from_res;}
//    else {*o = 0;} 
    *o = 0; 
}

/** \brief Extracts top node index. */
void static top_node(int* o, particle_template* p){
    std::string type = p -> type; 
    if (type == "jet"){
        jet* tx = (jet*)p; 
        if (tx -> top_index < 0){*o = -1;}
        else {*o = tx -> top_index;}
    }

    else if (type == "mu"){
        muon* tx = (muon*)p; 
        *o = tx -> top_index; 
    }

    else if (type == "el"){
        electron* tx = (electron*)p; 
        *o = tx -> top_index; 
    }
    else {*o = -1;} 
}; 

// --------------------- Node Observables --------------------- //
/** \brief Extracts transverse momentum. */
void static pt(double* o, particle_template* p){*o = p -> pt;} 

/** \brief Extracts pseudorapidity. */
void static eta(double* o, particle_template* p){*o = p -> eta;} 

/** \brief Extracts azimuthal angle. */
void static phi(double* o, particle_template* p){*o = p -> phi;} 

/** \brief Extracts energy. */
void static energy(double* o, particle_template* p){*o = p -> e;}

/** \brief Checks if the particle is a lepton. */
void static is_lepton(int* o, particle_template* p){*o = (p -> is_lep && !p -> is_nu);}

/** \brief Checks if the particle is a b-quark. */
void static is_bquark(int* o, particle_template* p){*o = p -> is_b;}

/** \brief Checks if the particle is a neutrino. */
void static is_neutrino(int* o, particle_template* p){*o = p -> is_nu;}

#endif
