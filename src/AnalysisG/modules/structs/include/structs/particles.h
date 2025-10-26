/**
 * @file particles.h
 * @brief Core particle data structure for AnalysisG framework
 * @defgroup modules_structs Core Data Structures
 * @{
 */

#ifndef PARTICLES_STRUCTS_H
#define PARTICLES_STRUCTS_H

#include <string>
#include <vector>
#include <map>

class particle_template;

/**
 * @struct particle_t
 * @brief Fundamental particle structure containing kinematic and identity information
 *
 * This structure represents a single particle (or physics object) in the AnalysisG framework.
 * It stores complete 4-momentum information in both Cartesian (px, py, pz, E) and polar
 * (pt, eta, phi, E) coordinates, along with particle identification and relationship data.
 *
 * ## Coordinate Systems
 *
 * The structure supports **dual representation**:
 * - **Cartesian**: (px, py, pz, E) - Standard 4-momentum components
 * - **Polar**: (pt, eta, phi, E) - ATLAS/CMS detector coordinates
 *
 * Flags `cartesian` and `polar` indicate which representation is currently valid.
 *
 * ## Usage Example
 * @code{.cpp}
 * particle_t electron;
 * electron.pt = 50.0;   // GeV
 * electron.eta = 0.5;
 * electron.phi = 1.2;
 * electron.e = 50.1;    // GeV
 * electron.polar = true;
 *
 * electron.pdgid = 11;  // PDG code for electron
 * electron.charge = -1.0;
 * electron.type = "Electron";
 * @endcode
 *
 * @note Memory layout is optimized for common access patterns in HEP analysis
 * @see EventTemplate for container of particles
 * @see particle_template for the complete particle interface
 */
struct particle_t {
    // ========== Energy and Mass ==========

    /**
     * @brief Particle energy in GeV
     *
     * Total energy E = √(p² + m²) where p is the 3-momentum magnitude.
     * Special sentinel value -0.000000000000001 indicates uninitialized state.
     *
     * @default -0.000000000000001 (uninitialized sentinel)
     * @unit GeV
     */
    double e = -0.000000000000001;

    /**
     * @brief Particle invariant mass in GeV
     *
     * Rest mass m where E² = p² + m². For massless particles (photons, gluons),
     * this should be 0. A negative value indicates the mass hasn't been set.
     *
     * @default -1.0 (uninitialized)
     * @unit GeV
     */
    double mass = -1;

    // ========== Cartesian 3-Momentum ==========

    /**
     * @brief x-component of momentum in GeV
     * @default 0.0
     * @unit GeV
     */
    double px = 0;

    /**
     * @brief y-component of momentum in GeV
     * @default 0.0
     * @unit GeV
     */
    double py = 0;

    /**
     * @brief z-component of momentum in GeV
     * @default 0.0
     * @unit GeV
     */
    double pz = 0;

    // ========== Polar Coordinates ==========

    /**
     * @brief Transverse momentum in GeV
     *
     * pt = √(px² + py²)
     * The momentum component perpendicular to the beam axis.
     *
     * @default 0.0
     * @unit GeV
     */
    double pt = 0;

    /**
     * @brief Pseudorapidity η (eta)
     *
     * η = -ln(tan(θ/2)) where θ is the polar angle from the beam axis.
     * Approximates rapidity for highly relativistic particles.
     *
     * @note η = 0 at 90° from beam, ±∞ along beam axis
     * @default 0.0
     * @unit dimensionless
     */
    double eta = 0;

    /**
     * @brief Azimuthal angle φ (phi) in radians
     *
     * Angle around the beam axis, typically in range [-π, π].
     *
     * @default 0.0
     * @unit radians
     */
    double phi = 0;

    // ========== Coordinate System Flags ==========

    /**
     * @brief Flag indicating if Cartesian coordinates (px, py, pz) are valid
     * @default false
     */
    bool cartesian = false;

    /**
     * @brief Flag indicating if polar coordinates (pt, eta, phi) are valid
     * @default false
     */
    bool polar = false;

    // ========== Particle Identity ==========

    /**
     * @brief Electric charge in units of elementary charge e
     *
     * Examples: electron = -1, proton = +1, neutron = 0
     *
     * @default 0.0
     * @unit e (elementary charge)
     */
    double charge = 0;

    /**
     * @brief PDG particle identification code
     *
     * Standard PDG Monte Carlo numbering scheme (see PDG Review).
     * Examples: 11 = e⁻, -11 = e⁺, 13 = μ⁻, 22 = γ, 2212 = p
     *
     * @see http://pdg.lbl.gov/
     * @default 0
     */
    int pdgid = 0;

    /**
     * @brief Index of particle in the event's particle collection
     *
     * Used for array indexing and relationship tracking.
     * A value of -1 indicates the particle is not in a collection.
     *
     * @default -1
     */
    int index = -1;

    // ========== Classification ==========

    /**
     * @brief Human-readable particle type/classification
     *
     * Examples: "Electron", "Muon", "Jet", "MET", "TruthTop"
     *
     * @default "" (empty string)
     */
    std::string type = "";

    /**
     * @brief Unique hash identifier for this particle
     *
     * Used for particle tracking across different event representations
     * and for efficient deduplication.
     *
     * @default "" (empty string)
     */
    std::string hash = "";

    /**
     * @brief Short symbol representation of the particle
     *
     * Examples: "e⁻", "μ⁺", "j", "ν"
     *
     * @default "" (empty string)
     */
    std::string symbol = "";

    // ========== Particle Type Definitions ==========

    /**
     * @brief PDG codes for charged leptons (e, μ, τ)
     *
     * Default includes: 11 (electron), 13 (muon), 15 (tau)
     * Used for lepton identification in analysis code.
     *
     * @default {11, 13, 15}
     */
    std::vector<int> lepdef = {11, 13, 15};

    /**
     * @brief PDG codes for neutrinos (νₑ, νᵤ, ντ)
     *
     * Default includes: 12 (e-neutrino), 14 (mu-neutrino), 16 (tau-neutrino)
     * Used for neutrino identification in analysis code.
     *
     * @default {12, 14, 16}
     */
    std::vector<int> nudef = {12, 14, 16};

    // ========== Decay Relationships ==========

    /**
     * @brief Map of child particle hashes
     *
     * Keys are hash identifiers of decay products, values indicate relationship validity.
     * Used to reconstruct decay chains and hierarchies.
     *
     * @note For truth particles, represents Monte Carlo truth decay tree
     * @note For detector objects, may represent jet constituents or composite objects
     *
     * @default {} (empty map)
     */
    std::map<std::string, bool> children = {};

    /**
     * @brief Map of parent particle hashes
     *
     * Keys are hash identifiers of parent particles, values indicate relationship validity.
     * Allows traversing decay chains upward from decay products to mothers.
     *
     * @default {} (empty map)
     */
    std::map<std::string, bool> parents = {};

    /**
     * @brief Pointer to the global particle data container
     *
     * Provides access to all particles in the event for relationship queries.
     * When non-null, hash values in children/parents maps can be dereferenced
     * to actual particle_template objects.
     *
     * @warning Raw pointer - does not manage lifetime. User must ensure validity.
     * @default nullptr
     */
    std::map<std::string, particle_template*>* data_p = nullptr;
};

/** @} */ // end of modules_structs group

#endif
