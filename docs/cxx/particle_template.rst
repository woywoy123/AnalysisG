.. cpp:class:: particle_template : public tools

    Base class representing a particle with kinematic properties and relationships.

    This class provides a template for particle objects used in physics analysis.
    It stores kinematic variables (like momentum, energy, mass) in both Cartesian
    (px, py, pz, e) and polar (pt, eta, phi, e) coordinates, handling conversions
    between them lazily. It also includes properties like PDG ID, charge, symbol,
    and allows tracking parent and child particles in a decay chain.

    The class utilizes a ``cproperty`` mechanism for accessing properties, allowing
    custom getter and setter logic (e.g., for coordinate transformations or
    on-demand calculations). It inherits from a ``tools`` base class (presumably
    providing utility functions like hashing).

    .. note::

        Coordinate systems are handled lazily. Accessing a variable in one system
        (e.g., ``pt``) might trigger an internal conversion if the data was last
        set or accessed in the other system (e.g., ``px``).

    :see: :cpp:class:`particle_t`, ``cproperty``, :cpp:class:`tools`

    .. cpp:function:: particle_template()

        Default constructor. Initializes properties with their respective getters/setters.

    .. cpp:function:: virtual ~particle_template()

        Virtual destructor. Handles cleanup, especially for dynamically allocated data
        related to particle trees (``data.data_p``) if used for serialization.

    .. cpp:function:: explicit particle_template(particle_t* p)

        Constructor initializing from a raw particle_t struct.

        :param p: Pointer to a particle_t struct containing initial data.

    .. cpp:function:: explicit particle_template(particle_template* p, bool dump = false)

        Copy-like constructor. Creates a new particle based on an existing one.

        :param p: Pointer to the source particle_template object.
        :param dump: If true, performs a deep copy of the particle's decay tree (parents/children)
                     potentially for serialization purposes (used by ``__reduce__``). If false,
                     only copies the immediate particle data, clearing parent/child links.

    .. cpp:function:: explicit particle_template(double px, double py, double pz, double e)

        Constructor initializing from Cartesian four-momentum components.

        :param px: The x-component of momentum.
        :param py: The y-component of momentum.
        :param pz: The z-component of momentum.
        :param e: The energy.

    .. cpp:function:: explicit particle_template(double px, double py, double pz)

        Constructor initializing from Cartesian three-momentum components.
        Energy is calculated on demand if needed.

        :param px: The x-component of momentum.
        :param py: The y-component of momentum.
        :param pz: The z-component of momentum.

    .. cpp:function:: void to_cartesian()

        Ensures the internal representation is in Cartesian coordinates (px, py, pz).
        Performs the conversion from polar coordinates (pt, eta, phi) if necessary.

    .. cpp:function:: void to_polar()

        Ensures the internal representation is in polar coordinates (pt, eta, phi).
        Performs the conversion from Cartesian coordinates (px, py, pz) if necessary.

    .. rubric:: Properties

    .. cpp:member:: cproperty<double, particle_template> e

        Property representing the particle's energy (E).
        Access triggers :cpp:func:`get_e` or :cpp:func:`set_e`.

        :see: :cpp:func:`get_e`, :cpp:func:`set_e`

    .. cpp:function:: static void set_e(double*, particle_template*)

        Setter function for the energy property.

    .. cpp:function:: static void get_e(double*, particle_template*)

        Getter function for the energy property. Calculates from momentum and mass if not set.

    .. cpp:member:: cproperty<double, particle_template> mass

        Property representing the particle's invariant mass (m).
        Access triggers :cpp:func:`get_mass` or :cpp:func:`set_mass`.

        :see: :cpp:func:`get_mass`, :cpp:func:`set_mass`

    .. cpp:function:: static void set_mass(double*, particle_template*)

        Setter function for the mass property.

    .. cpp:function:: static void get_mass(double*, particle_template*)

        Getter function for the mass property. Calculates from four-momentum if not set.

    .. cpp:member:: cproperty<double, particle_template> pt

        Property representing the particle's transverse momentum (pT).
        Access triggers :cpp:func:`get_pt` or :cpp:func:`set_pt`. Requires polar coordinates.

        :see: :cpp:func:`get_pt`, :cpp:func:`set_pt`, :cpp:func:`to_polar`

    .. cpp:function:: static void set_pt(double*, particle_template*)

        Setter function for the pT property. Marks internal state as needing Cartesian update.

    .. cpp:function:: static void get_pt(double*, particle_template*)

        Getter function for the pT property. Ensures polar coordinates before returning.

    .. cpp:member:: cproperty<double, particle_template> eta

        Property representing the particle's pseudorapidity (eta).
        Access triggers :cpp:func:`get_eta` or :cpp:func:`set_eta`. Requires polar coordinates.

        :see: :cpp:func:`get_eta`, :cpp:func:`set_eta`, :cpp:func:`to_polar`

    .. cpp:function:: static void set_eta(double*, particle_template*)

        Setter function for the eta property. Marks internal state as needing Cartesian update.

    .. cpp:function:: static void get_eta(double*, particle_template*)

        Getter function for the eta property. Ensures polar coordinates before returning.

    .. cpp:member:: cproperty<double, particle_template> phi

        Property representing the particle's azimuthal angle (phi).
        Access triggers :cpp:func:`get_phi` or :cpp:func:`set_phi`. Requires polar coordinates.

        :see: :cpp:func:`get_phi`, :cpp:func:`set_phi`, :cpp:func:`to_polar`

    .. cpp:function:: static void set_phi(double*, particle_template*)

        Setter function for the phi property. Marks internal state as needing Cartesian update.

    .. cpp:function:: static void get_phi(double*, particle_template*)

        Getter function for the phi property. Ensures polar coordinates before returning.

    .. cpp:member:: cproperty<double, particle_template> px

        Property representing the particle's x-component of momentum (px).
        Access triggers :cpp:func:`get_px` or :cpp:func:`set_px`. Requires Cartesian coordinates.

        :see: :cpp:func:`get_px`, :cpp:func:`set_px`, :cpp:func:`to_cartesian`

    .. cpp:function:: static void set_px(double*, particle_template*)

        Setter function for the px property. Marks internal state as needing polar update.

    .. cpp:function:: static void get_px(double*, particle_template*)

        Getter function for the px property. Ensures Cartesian coordinates before returning.

    .. cpp:member:: cproperty<double, particle_template> py

        Property representing the particle's y-component of momentum (py).
        Access triggers :cpp:func:`get_py` or :cpp:func:`set_py`. Requires Cartesian coordinates.

        :see: :cpp:func:`get_py`, :cpp:func:`set_py`, :cpp:func:`to_cartesian`

    .. cpp:function:: static void set_py(double*, particle_template*)

        Setter function for the py property. Marks internal state as needing polar update.

    .. cpp:function:: static void get_py(double*, particle_template*)

        Getter function for the py property. Ensures Cartesian coordinates before returning.

    .. cpp:member:: cproperty<double, particle_template> pz

        Property representing the particle's z-component of momentum (pz).
        Access triggers :cpp:func:`get_pz` or :cpp:func:`set_pz`. Requires Cartesian coordinates.

        :see: :cpp:func:`get_pz`, :cpp:func:`set_pz`, :cpp:func:`to_cartesian`

    .. cpp:function:: static void set_pz(double*, particle_template*)

        Setter function for the pz property. Marks internal state as needing polar update.

    .. cpp:function:: static void get_pz(double*, particle_template*)

        Getter function for the pz property. Ensures Cartesian coordinates before returning.

    .. cpp:member:: cproperty<int, particle_template> pdgid

        Property representing the particle's Particle Data Group ID (PDG ID).
        Access triggers :cpp:func:`get_pdgid` or :cpp:func:`set_pdgid`.

        :see: :cpp:func:`get_pdgid`, :cpp:func:`set_pdgid`

    .. cpp:function:: static void set_pdgid(int*, particle_template*)

        Setter function for the PDG ID property.

    .. cpp:function:: static void get_pdgid(int*, particle_template*)

        Getter function for the PDG ID property. Can infer from symbol if PDG ID is 0.

    .. cpp:member:: cproperty<std::string, particle_template> symbol

        Property representing the particle's symbol (e.g., "e", "mu", "$\\gamma$").
        Access triggers :cpp:func:`get_symbol` or :cpp:func:`set_symbol`.

        :see: :cpp:func:`get_symbol`, :cpp:func:`set_symbol`

    .. cpp:function:: static void set_symbol(std::string*, particle_template*)

        Setter function for the symbol property.

    .. cpp:function:: static void get_symbol(std::string*, particle_template*)

        Getter function for the symbol property. Can infer from PDG ID if symbol is empty.

    .. cpp:member:: cproperty<double, particle_template> charge

        Property representing the particle's electric charge.
        Access triggers :cpp:func:`get_charge` or :cpp:func:`set_charge`.

        :see: :cpp:func:`get_charge`, :cpp:func:`set_charge`

    .. cpp:function:: static void set_charge(double*, particle_template*)

        Setter function for the charge property.

    .. cpp:function:: static void get_charge(double*, particle_template*)

        Getter function for the charge property.

    .. cpp:member:: cproperty<std::string, particle_template> hash

        Read-only property representing a unique hash identifier for the particle.
        Calculated based on the Cartesian four-momentum. Access triggers :cpp:func:`get_hash`.

        :see: :cpp:func:`get_hash`

    .. cpp:function:: static void get_hash(std::string*, particle_template*)

        Getter function for the hash property. Calculates hash from Cartesian four-momentum if not already computed.

    .. cpp:member:: cproperty<bool, particle_template> is_b

        Read-only property indicating if the particle is a b-quark (PDG ID +/- 5).
        Access triggers :cpp:func:`get_isb`.

        :see: :cpp:func:`get_isb`, :cpp:func:`is`

    .. cpp:function:: static void get_isb(bool*, particle_template*)

        Getter function for the is_b property. Uses ``is({5})``.

    .. cpp:member:: cproperty<bool, particle_template> is_lep

        Read-only property indicating if the particle is a lepton (e, mu, tau, or their neutrinos, based on ``data.lepdef``).
        Access triggers :cpp:func:`get_islep`.

        :see: :cpp:func:`get_islep`, :cpp:func:`is`, :cpp:member:`particle_t::lepdef`

    .. cpp:function:: static void get_islep(bool*, particle_template*)

        Getter function for the is_lep property. Uses ``is(data.lepdef)``.

    .. cpp:member:: cproperty<bool, particle_template> is_nu

        Read-only property indicating if the particle is a neutrino (based on ``data.nudef``).
        Access triggers :cpp:func:`get_isnu`.

        :see: :cpp:func:`get_isnu`, :cpp:func:`is`, :cpp:member:`particle_t::nudef`

    .. cpp:function:: static void get_isnu(bool*, particle_template*)

        Getter function for the is_nu property. Uses ``is(data.nudef)``.

    .. cpp:member:: cproperty<bool, particle_template> is_add

        Read-only property indicating if the particle is *not* a b-quark, lepton, or neutrino.
        Access triggers :cpp:func:`get_isadd`.

        :see: :cpp:func:`get_isadd`, :cpp:member:`is_b`, :cpp:member:`is_lep`, :cpp:member:`is_nu`

    .. cpp:function:: static void get_isadd(bool*, particle_template*)

        Getter function for the is_add property. Returns ``!(is_lep || is_nu || is_b)``.

    .. cpp:member:: cproperty<bool, particle_template> lep_decay

        Read-only property indicating if the particle has both a lepton and a neutrino among its direct children.
        Access triggers :cpp:func:`get_lepdecay`.

        :see: :cpp:func:`get_lepdecay`, :cpp:member:`children`, :cpp:member:`is_lep`, :cpp:member:`is_nu`

    .. cpp:function:: static void get_lepdecay(bool*, particle_template*)

        Getter function for the lep_decay property. Checks children for leptons and neutrinos.

    .. cpp:member:: cproperty<std::map<std::string, particle_template*>, particle_template> parents

        Property representing the map of parent particles (key: hash, value: particle pointer).
        Access triggers :cpp:func:`get_parents` or :cpp:func:`set_parents`.

        :see: :cpp:func:`get_parents`, :cpp:func:`set_parents`, :cpp:func:`register_parent`, :cpp:member:`m_parents`

    .. cpp:function:: static void set_parents(std::map<std::string, particle_template*>*, particle_template*)

        Setter function for the parents property. Registers each particle in the input map as a parent. Clears parents if map is empty.

    .. cpp:function:: static void get_parents(std::map<std::string, particle_template*>*, particle_template*)

        Getter function for the parents property. Returns the internal map :cpp:member:`m_parents`.

    .. cpp:member:: cproperty<std::map<std::string, particle_template*>, particle_template> children

        Property representing the map of child particles (key: hash, value: particle pointer).
        Access triggers :cpp:func:`get_children` or :cpp:func:`set_children`.

        :see: :cpp:func:`get_children`, :cpp:func:`set_children`, :cpp:func:`register_child`, :cpp:member:`m_children`

    .. cpp:function:: static void set_children(std::map<std::string, particle_template*>*, particle_template*)

        Setter function for the children property. Registers each particle in the input map as a child. Clears children if map is empty.

    .. cpp:function:: static void get_children(std::map<std::string, particle_template*>*, particle_template*)

        Getter function for the children property. Returns the internal map :cpp:member:`m_children`.

    .. cpp:member:: cproperty<std::string, particle_template> type

        Property representing a type identifier string for the particle (e.g., "Jet", "Electron").
        Used for categorization or naming conventions. Access triggers :cpp:func:`get_type` or :cpp:func:`set_type`.

        :see: :cpp:func:`get_type`, :cpp:func:`set_type`, :cpp:func:`apply_type_prefix`

    .. cpp:function:: static void set_type(std::string*, particle_template*)

        Setter function for the type property.

    .. cpp:function:: static void get_type(std::string*, particle_template*)

        Getter function for the type property.

    .. cpp:member:: cproperty<int, particle_template> index

        Property representing an index, potentially its position within a collection.
        Access triggers :cpp:func:`get_index` or :cpp:func:`set_index`.

        :see: :cpp:func:`get_index`, :cpp:func:`set_index`

    .. cpp:function:: static void set_index(int*, particle_template*)

        Setter function for the index property.

    .. cpp:function:: static void get_index(int*, particle_template*)

        Getter function for the index property.

    .. rubric:: Methods

    .. cpp:function:: bool is(std::vector<int> p)

        Checks if the particle's PDG ID matches any ID in the provided list (absolute values compared).

        :param p: Vector of PDG IDs to check against.
        :return: True if the absolute value of the particle's PDG ID matches any absolute value in the list, false otherwise.

    .. cpp:function:: double DeltaR(particle_template* p)

        Calculates the angular separation Delta R = sqrt( (delta Eta)^2 + (delta Phi)^2 ) between this particle and another.

        :param p: Pointer to the other particle_template object.
        :return: The Delta R value.
        :note: Handles phi wraparound correctly.

    .. cpp:function:: bool operator==(particle_template& p)

        Equality comparison operator. Compares particles based on their hash values.

        :param p: The particle_template object to compare against.
        :return: True if the hash values are identical, false otherwise.
        :see: :cpp:member:`hash`

    .. cpp:function:: template <typename g> g operator+(g& p)

        Addition operator. Adds the four-vectors of this particle and another.

        :tparam g: The type of the particle to add (must have px, py, pz, e properties/members accessible as doubles).
        :param p: The particle object to add.
        :return: A new particle object (``g``) representing the sum of the four-vectors.
                 The type and polar flag are set based on the current particle.
        :note: The returned particle's internal state is marked as needing polar update.

    .. cpp:function:: void operator+=(particle_template* p)

        In-place addition operator. Adds the four-vector of another particle to this one.

        :param p: Pointer to the particle_template object to add.
        :note: Ensures both particles are in Cartesian coordinates before adding. Marks internal state as needing polar update.
        :see: :cpp:func:`iadd`

    .. cpp:function:: void iadd(particle_template* p)

        In-place addition method (alternative syntax for ``operator+=``).

        :param p: Pointer to the particle_template object to add.
        :see: :cpp:func:`operator+=`

    .. cpp:function:: bool register_parent(particle_template* p)

        Registers a particle as a parent of this particle.
        Adds the parent to the internal :cpp:member:`m_parents` map and updates the ``data.parents`` map.

        :param p: Pointer to the parent particle_template object.
        :return: True if the parent was successfully registered (or already existed), false otherwise (should always return true currently).
        :see: :cpp:member:`parents`, :cpp:member:`m_parents`, :cpp:member:`data`

    .. cpp:member:: std::map<std::string, particle_template*> m_parents

        Internal map storing pointers to parent particles, keyed by hash.

    .. cpp:function:: bool register_child(particle_template* p)

        Registers a particle as a child of this particle.
        Adds the child to the internal :cpp:member:`m_children` map and updates the ``data.children`` map.

        :param p: Pointer to the child particle_template object.
        :return: True if the child was newly registered, false if it already existed.
        :see: :cpp:member:`children`, :cpp:member:`m_children`, :cpp:member:`data`

    .. cpp:member:: std::map<std::string, particle_template*> m_children

        Internal map storing pointers to child particles, keyed by hash.

    .. cpp:function:: void add_leaf(std::string key, std::string leaf = "")

        Adds a key-value pair to the :cpp:member:`leaves` map, used for naming output branches/variables.

        :param key: The internal name or identifier for the leaf.
        :param leaf: The desired output name for the leaf. If empty, defaults to ``key``.
        :see: :cpp:member:`leaves`, :cpp:func:`apply_type_prefix`

    .. cpp:member:: std::map<std::string, std::string> leaves

        Map storing leaf names for output, mapping internal keys to output names.

    .. cpp:function:: void apply_type_prefix()

        Prepends the particle's :cpp:member:`type` string to all values in the :cpp:member:`leaves` map.
        Useful for creating unique branch names (e.g., "Jet_pt" instead of just "pt").

        :see: :cpp:member:`leaves`, :cpp:member:`type`

    .. cpp:function:: std::map<std::string, std::map<std::string, particle_t>> __reduce__()

        Method likely used for serialization or reducing the particle and its tree for storage/transfer (e.g., to Python via ROOT).
        Creates a deep copy of the particle and its connected parents/children (those reachable via the ``dump=true`` constructor logic)
        and returns a map containing their underlying ``particle_t`` data, keyed by hash.

        :return: A map where keys are particle hashes and values are maps containing the ``particle_t`` data under the key "data".
        :note: Uses the ``dump=true`` constructor and internal ``data.data_p`` pointer for tracking during the deep copy. Skips particles marked with :cpp:member:`_is_serial`.

    .. cpp:function:: virtual void build(std::map<std::string, particle_template*>* event, element_t* el)

        Virtual method intended for derived classes to implement specific building logic.
        Potentially used to populate particle properties based on event data or other elements.

        :param event: Pointer to a map representing the current event's particles (unused in base).
        :param el: Pointer to an element_t struct (unused in base).

    .. cpp:function:: virtual particle_template* clone()

        Virtual method for creating a clone (copy) of the particle object.

        :return: A pointer to a new particle_template object (base implementation returns a default-constructed one).
        :note: Derived classes should override this to return a copy of their specific type.

    .. cpp:member:: particle_t data

        The underlying data structure holding the particle's properties.

        :see: :cpp:class:`particle_t`

    .. cpp:member:: bool _is_serial = false

        Internal flag, likely used during serialization (``__reduce__``) to avoid infinite loops or redundant processing.

    .. cpp:member:: bool _is_marked = false

        Internal flag, potentially used for marking particles during traversal algorithms.

.. note::

   The following C++ code snippets illustrate the header definitions and some implementation details. For full context, refer to the source files.

.. code-block:: cpp
   :caption: Header Snippet (particle_template.h)

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
    #include <vector>
    #include <map>

    class event_template;
    class selection_template;

    class particle_template : public tools
    {
    public:
        particle_template();
        virtual ~particle_template();

        explicit particle_template(particle_t* p);
        explicit particle_template(particle_template* p, bool dump = false);
        explicit particle_template(double px, double py, double pz, double e);
        explicit particle_template(double px, double py, double pz);

        void to_cartesian();
        void to_polar();

        // --- Properties ---
        cproperty<double, particle_template> e;
        void static set_e(double*, particle_template*);
        void static get_e(double*, particle_template*);

        cproperty<double, particle_template> mass;
        void static set_mass(double*, particle_template*);
        void static get_mass(double*, particle_template*);

        cproperty<double, particle_template> pt;
        void static set_pt(double*, particle_template*);
        void static get_pt(double*, particle_template*);

        cproperty<double, particle_template> eta;
        void static set_eta(double*, particle_template*);
        void static get_eta(double*, particle_template*);

        cproperty<double, particle_template> phi;
        void static set_phi(double*, particle_template*);
        void static get_phi(double*, particle_template*);

        cproperty<double, particle_template> px;
        void static set_px(double*, particle_template*);
        void static get_px(double*, particle_template*);

        cproperty<double, particle_template> py;
        void static set_py(double*, particle_template*);
        void static get_py(double*, particle_template*);

        cproperty<double, particle_template> pz;
        void static set_pz(double*, particle_template*);
        void static get_pz(double*, particle_template*);

        cproperty<int, particle_template> pdgid;
        void static set_pdgid(int*, particle_template*);
        void static get_pdgid(int*, particle_template*);

        cproperty<std::string, particle_template> symbol;
        void static set_symbol(std::string*, particle_template*);
        void static get_symbol(std::string*, particle_template*);

        cproperty<double, particle_template> charge;
        void static set_charge(double*, particle_template*);
        void static get_charge(double*, particle_template*);

        cproperty<std::string, particle_template> hash;
        void static get_hash(std::string*, particle_template*);

        bool is(std::vector<int> p);
        cproperty<bool, particle_template> is_b;
        void static get_isb(bool*, particle_template*);

        cproperty<bool, particle_template> is_lep;
        void static get_islep(bool*, particle_template*);

        cproperty<bool, particle_template> is_nu;
        void static get_isnu(bool*, particle_template*);

        cproperty<bool, particle_template> is_add;
        void static get_isadd(bool*, particle_template*);

        cproperty<bool, particle_template> lep_decay;
        void static get_lepdecay(bool*, particle_template*);

        cproperty<std::map<std::string, particle_template*>, particle_template> parents;
        void static set_parents(std::map<std::string, particle_template*>*, particle_template*);
        void static get_parents(std::map<std::string, particle_template*>*, particle_template*);

        cproperty<std::map<std::string, particle_template*>, particle_template> children;
        void static set_children(std::map<std::string, particle_template*>*, particle_template*);
        void static get_children(std::map<std::string, particle_template*>*, particle_template*);

        cproperty<std::string, particle_template> type;
        void static set_type(std::string*, particle_template*);
        void static get_type(std::string*, particle_template*);

        cproperty<int, particle_template> index;
        void static set_index(int*, particle_template*);
        void static get_index(int*, particle_template*);

        // --- Methods ---
        double DeltaR(particle_template* p);

        bool operator == (particle_template& p);

        template <typename g>
        g operator + (g& p){
            g p2 = g();
            p2.data.px = double(p.px) + double(this -> px);
            p2.data.py = double(p.py) + double(this -> py);
            p2.data.pz = double(p.pz) + double(this -> pz);
            p2.data.e  = double(p.e ) + double(this -> e);
            p2.data.type = this -> data.type;
            p2.data.polar = true; // Mark as needing polar update
            return p2;
        }

        void operator += (particle_template* p);
        void iadd(particle_template* p);

        bool register_parent(particle_template* p);
        std::map<std::string, particle_template*> m_parents;

        bool register_child(particle_template* p);
        std::map<std::string, particle_template*> m_children;

        void add_leaf(std::string key, std::string leaf = "");
        std::map<std::string, std::string> leaves = {};

        void apply_type_prefix();
        std::map<std::string, std::map<std::string, particle_t>> __reduce__();

        virtual void build(std::map<std::string, particle_template*>* event, element_t* el);
        virtual particle_template* clone();

        particle_t data;

        bool _is_serial = false;
        bool _is_marked = false;
    };
    #endif

.. code-block:: cpp
   :caption: Implementation Snippet (Example: Cartesian Getters/Setters)

    #include <templates/particle_template.h>
    #include <cmath> // For std::cos, std::sin, std::sinh

    void particle_template::set_px(double* val, particle_template* prt){
        prt -> data.px = *val;
        prt -> data.polar = true; // Mark as needing polar update
    }

    void particle_template::get_px(double* val, particle_template* prt){
        prt -> to_cartesian();
        *val = prt -> data.px;
    }

    void particle_template::set_py(double* val, particle_template* prt){
        prt -> data.py = *val;
        prt -> data.polar = true; // Mark as needing polar update
    }

    void particle_template::get_py(double* val, particle_template* prt){
        prt -> to_cartesian();
        *val = prt -> data.py;
    }

    void particle_template::set_pz(double* val, particle_template* prt){
        prt -> data.pz = *val;
        prt -> data.polar = true; // Mark as needing polar update
    }

    void particle_template::get_pz(double* val, particle_template* prt){
        prt -> to_cartesian();
        *val = prt -> data.pz;
    }

    void particle_template::to_cartesian(){
        particle_t* p = &this -> data;
        if (!p -> cartesian){ return; }
