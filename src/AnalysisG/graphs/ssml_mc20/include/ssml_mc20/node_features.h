/**
 * @file node_features.h
 * @brief Defines node feature extraction functions for SSML MC20 graphs.
 *
 * This file contains function definitions for extracting various features from
 * particle data to create node representations in graph neural networks specific
 * to the SSML MC20 analysis. It includes both truth-level functions for training/validation
 * and observable-level functions for inference on real data.
 */

#ifndef NODES_FEATURES_SSML_MC20_H ///< Start of include guard for NODES_FEATURES_SSML_MC20_H.
#define NODES_FEATURES_SSML_MC20_H ///< Definition of NODES_FEATURES_SSML_MC20_H to signify the header has been included.

// --------------------- Node Truth Features --------------------- //
/**
 * @brief Extracts resonance information from a particle.
 * @param o Pointer to int where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Determines if the particle originates from a resonance decay and stores
 * the appropriate classification value.
 */
void static res_node(int* o, particle_template* p){
    std::string type = p -> type;
    if (type == "mu"){*o = ((muon*)p) -> from_res;}
    else if (type == "el"){*o = ((electron*)p) -> from_res;}
    else if (type == "jet"){*o = ((jet*)p) -> from_res;}
    else {*o = 0;} 
}; 

/**
 * @brief Extracts top quark origin information from a particle.
 * @param o Pointer to int where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Determines if the particle originates from a top quark decay and stores
 * the appropriate classification value.
 */
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

// --------------------- Node Observable Features --------------------- //
/**
 * @brief Extracts transverse momentum feature from a particle.
 * @param o Pointer to double where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Calculates and stores the transverse momentum (pT) of the particle.
 */
void static pt(double* o, particle_template* p){*o = p -> pt;} 

/**
 * @brief Extracts pseudorapidity feature from a particle.
 * @param o Pointer to double where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Calculates and stores the pseudorapidity (eta) of the particle.
 */
void static eta(double* o, particle_template* p){*o = p -> eta;} 

/**
 * @brief Extracts azimuthal angle feature from a particle.
 * @param o Pointer to double where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Calculates and stores the azimuthal angle (phi) of the particle.
 */
void static phi(double* o, particle_template* p){*o = p -> phi;} 

/**
 * @brief Extracts energy feature from a particle.
 * @param o Pointer to double where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Retrieves and stores the energy of the particle.
 */
void static energy(double* o, particle_template* p){*o = p -> e;}

/**
 * @brief Determines if a particle is a lepton.
 * @param o Pointer to int where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Sets the output to 1 if the particle is an electron or muon, 0 otherwise.
 */
void static is_lepton(int* o, particle_template* p){*o = (p -> is_lep && !p -> is_nu);}

/**
 * @brief Determines if a particle is a b-quark.
 * @param o Pointer to int where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Sets the output to 1 if the particle is a b-quark, 0 otherwise.
 */
void static is_bquark(int* o, particle_template* p){*o = p -> is_b;}

/**
 * @brief Determines if a particle is a neutrino.
 * @param o Pointer to int where the output will be stored.
 * @param p Pointer to the particle_template containing the data.
 *
 * Sets the output to 1 if the particle is a neutrino, 0 otherwise.
 */
void static is_neutrino(int* o, particle_template* p){*o = p -> is_nu;}


#endif ///< End of include guard for NODES_FEATURES_SSML_MC20_H.
