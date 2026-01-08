/**
 * @file particle_template.h
 * @brief Defines the particle_template base class for physics particle representations.
 *
 * This file contains the declaration of the `particle_template` class, which serves
 * as the base class for all particle types in the AnalysisG framework. It provides
 * common functionality for four-momentum handling, particle identification, and
 * relationship tracking (parents/children).
 */

#ifndef PARTICLETEMPLATE_H
#define PARTICLETEMPLATE_H

#include <structs/particles.h>
#include <structs/property.h>
#include <structs/element.h>
#include <tools/tools.h>

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cmath>

class event_template; 
class selection_template; 

/**
 * @class particle_template
 * @brief Base class for representing physics particles with four-momentum and properties.
 *
 * The particle_template class provides a comprehensive interface for working with
 * physics particles. It supports:
 * - Four-momentum in both Cartesian (px, py, pz, E) and polar (pT, η, φ, E) representations
 * - Particle identification (PDG ID, symbol, charge)
 * - Parent/child relationship tracking for decay chains
 * - Arithmetic operations for combining particles
 * - DeltaR calculations for angular separation
 *
 * @section particle_usage Usage
 *
 * Subclass particle_template to create specific particle types:
 *
 * ```cpp
 * class Electron : public particle_template {
 * public:
 *     double isolation;
 *     bool passID;
 *     
 *     Electron() : particle_template() {
 *         add_leaf("isolation", "el_iso");
 *         add_leaf("passID", "el_tight");
 *     }
 *     
 *     Electron* clone() override {
 *         return new Electron(*this);
 *     }
 * };
 * ```
 *
 * @section particle_kinematics Kinematic Properties
 *
 * Access particle kinematics through properties:
 * - `pt`: Transverse momentum
 * - `eta`: Pseudorapidity
 * - `phi`: Azimuthal angle
 * - `e`: Energy
 * - `mass`: Invariant mass
 * - `px`, `py`, `pz`: Cartesian momentum components
 *
 * @section particle_id Identification Properties
 *
 * - `pdgid`: PDG particle ID
 * - `symbol`: Particle symbol (e.g., "e-", "μ+")
 * - `charge`: Electric charge
 * - `is_b`, `is_lep`, `is_nu`, `is_add`: Boolean type flags
 */
class particle_template : public tools
{
    public:
        /**
         * @brief Default constructor.
         * Initializes a particle with zero four-momentum.
         */
        particle_template();
        
        /**
         * @brief Virtual destructor.
         * Cleans up resources including parent/child references.
         */
        virtual ~particle_template(); 

        /**
         * @brief Constructs a particle from a particle_t data structure.
         * @param p Pointer to the particle_t structure containing particle data.
         */
        explicit particle_template(particle_t* p);
        
        /**
         * @brief Copy constructor from another particle_template.
         * @param p Pointer to the particle to copy.
         * @param dump If true, also copies parent/child relationships.
         */
        explicit particle_template(particle_template* p, bool dump = false); 
        
        /**
         * @brief Constructs a particle with specified Cartesian four-momentum.
         * @param px X-component of momentum.
         * @param py Y-component of momentum.
         * @param pz Z-component of momentum.
         * @param e Energy.
         */
        explicit particle_template(double px, double py, double pz, double e); 
        
        /**
         * @brief Constructs a massless particle with specified Cartesian three-momentum.
         * @param px X-component of momentum.
         * @param py Y-component of momentum.
         * @param pz Z-component of momentum.
         */
        explicit particle_template(double px, double py, double pz);

        /**
         * @brief Converts internal representation to Cartesian coordinates.
         */
        void to_cartesian(); 
        
        /**
         * @brief Converts internal representation to polar coordinates.
         */
        void to_polar(); 

        cproperty<double, particle_template> e;    ///< Energy property.
        void static set_e(double*, particle_template*);    ///< Setter for energy.
        void static get_e(double*, particle_template*);    ///< Getter for energy.

        cproperty<double, particle_template> mass; ///< Invariant mass property.
        void static set_mass(double*, particle_template*); ///< Setter for mass.
        void static get_mass(double*, particle_template*); ///< Getter for mass.

        cproperty<double, particle_template> pt;   ///< Transverse momentum property.
        void static set_pt(double*, particle_template*);   ///< Setter for pT.
        void static get_pt(double*, particle_template*);   ///< Getter for pT.

        cproperty<double, particle_template> eta;  ///< Pseudorapidity property.
        void static set_eta(double*, particle_template*);  ///< Setter for η.
        void static get_eta(double*, particle_template*);  ///< Getter for η.

        cproperty<double, particle_template> phi;  ///< Azimuthal angle property.
        void static set_phi(double*, particle_template*);  ///< Setter for φ.
        void static get_phi(double*, particle_template*);  ///< Getter for φ.

        cproperty<double, particle_template> px;   ///< X-momentum property.
        void static set_px(double*, particle_template*);   ///< Setter for px.
        void static get_px(double*, particle_template*);   ///< Getter for px.

        cproperty<double, particle_template> py;   ///< Y-momentum property.
        void static set_py(double*, particle_template*);   ///< Setter for py.
        void static get_py(double*, particle_template*);   ///< Getter for py.

        cproperty<double, particle_template> pz;   ///< Z-momentum property.
        void static set_pz(double*, particle_template*);   ///< Setter for pz.
        void static get_pz(double*, particle_template*);   ///< Getter for pz.

        cproperty<int, particle_template> pdgid;   ///< PDG particle ID property.
        void static set_pdgid(int*, particle_template*);   ///< Setter for PDG ID.
        void static get_pdgid(int*, particle_template*);   ///< Getter for PDG ID.

        cproperty<std::string, particle_template> symbol; ///< Particle symbol property.
        void static set_symbol(std::string*, particle_template*); ///< Setter for symbol.
        void static get_symbol(std::string*, particle_template*); ///< Getter for symbol.

        cproperty<double, particle_template> charge; ///< Electric charge property.
        void static set_charge(double*, particle_template*); ///< Setter for charge.
        void static get_charge(double*, particle_template*); ///< Getter for charge.

        cproperty<std::string, particle_template> hash; ///< Unique hash identifier.
        void static get_hash(std::string*, particle_template*); ///< Getter for hash.

        /**
         * @brief Checks if particle matches any of the given PDG IDs.
         * @param p Vector of PDG IDs to check against.
         * @return True if particle's PDG ID matches any in the vector.
         */
        bool is(std::vector<int> p); 
        
        cproperty<bool, particle_template> is_b;   ///< Is b-quark/hadron flag.
        void static get_isb(bool*, particle_template*);    ///< Getter for is_b.

        cproperty<bool, particle_template> is_lep; ///< Is lepton flag.
        void static get_islep(bool*, particle_template*);  ///< Getter for is_lep.

        cproperty<bool, particle_template> is_nu;  ///< Is neutrino flag.
        void static get_isnu(bool*, particle_template*);   ///< Getter for is_nu.

        cproperty<bool, particle_template> is_add; ///< Is additional radiation flag.
        void static get_isadd(bool*, particle_template*);  ///< Getter for is_add.

        cproperty<bool, particle_template> lep_decay; ///< Leptonic decay flag.
        void static get_lepdecay(bool*, particle_template*); ///< Getter for lep_decay.

        cproperty<std::map<std::string, particle_template*>, particle_template> parents; ///< Parent particles.
        void static set_parents(std::map<std::string, particle_template*>*, particle_template*); ///< Setter for parents.
        void static get_parents(std::map<std::string, particle_template*>*, particle_template*); ///< Getter for parents.

        cproperty<std::map<std::string, particle_template*>, particle_template> children; ///< Child particles.
        void static set_children(std::map<std::string, particle_template*>*, particle_template*); ///< Setter for children.
        void static get_children(std::map<std::string, particle_template*>*, particle_template*); ///< Getter for children.

        cproperty<std::string, particle_template> type; ///< Particle type string.
        void static set_type(std::string*, particle_template*); ///< Setter for type.
        void static get_type(std::string*, particle_template*); ///< Getter for type.

        cproperty<int, particle_template> index;   ///< Index in particle collection.
        void static set_index(int*, particle_template*);   ///< Setter for index.
        void static get_index(int*, particle_template*);   ///< Getter for index.

        /**
         * @brief Calculates the angular separation ΔR from another particle.
         * @param p Pointer to the other particle.
         * @return ΔR = √(Δη² + Δφ²)
         */
        double DeltaR(particle_template* p);

        /**
         * @brief Equality comparison operator.
         * @param p The particle to compare against.
         * @return True if particles have the same hash.
         */
        bool operator == (particle_template& p); 

        /**
         * @brief Addition operator for combining particles.
         * @tparam g The particle type (must inherit from particle_template).
         * @param p The particle to add.
         * @return A new particle with combined four-momentum.
         */
        template <typename g>
        g operator + (g& p){
            g p2 = g(); 
            p2.data.px = double(p.px) + double(this -> px); 
            p2.data.py = double(p.py) + double(this -> py);  
            p2.data.pz = double(p.pz) + double(this -> pz); 
            p2.data.e  = double(p.e ) + double(this -> e); 
            p2.data.type = this -> data.type; 
            p2.data.polar = true; 
            return p2; 
        }

        /**
         * @brief In-place addition operator.
         * @param p Pointer to the particle to add.
         */
        void operator += (particle_template* p); 
        
        /**
         * @brief In-place addition method (alternative to +=).
         * @param p Pointer to the particle to add.
         */
        void iadd(particle_template* p); 
      
        /**
         * @brief Registers a parent particle.
         * @param p Pointer to the parent particle.
         * @return True if registration was successful.
         */
        bool register_parent(particle_template* p);
        std::map<std::string, particle_template*> m_parents; ///< Map of parent particles by hash.

        /**
         * @brief Registers a child particle.
         * @param p Pointer to the child particle.
         * @return True if registration was successful.
         */
        bool register_child(particle_template* p);
        std::map<std::string, particle_template*> m_children; ///< Map of child particles by hash.

        /**
         * @brief Adds a leaf (variable) to read for this particle.
         * @param key The internal name for this leaf.
         * @param leaf The actual leaf name in the ROOT file (optional, defaults to key).
         */
        void add_leaf(std::string key, std::string leaf = ""); 
        std::map<std::string, std::string> leaves = {}; ///< Map of leaf names.

        /**
         * @brief Applies the particle type as a prefix to leaf names.
         */
        void apply_type_prefix(); 
        
        /**
         * @brief Serializes particle data for storage.
         * @return Nested map structure for serialization.
         */
        std::map<std::string, std::map<std::string, particle_t>> __reduce__(); 

        /**
         * @brief Builds the particle from event data.
         * @param event Map of particles in the event.
         * @param el Element containing raw data.
         */
        virtual void build(std::map<std::string, particle_template*>* event, element_t* el); 
        
        /**
         * @brief Creates a clone of this particle.
         * @return Pointer to the cloned particle.
         */
        virtual particle_template* clone(); 
        
        particle_t data;  ///< Internal data structure holding particle properties.

        bool _is_serial = false; ///< Flag indicating if particle is being serialized.
        bool _is_marked = false; ///< Flag for garbage collection marking.
}; 
#endif

