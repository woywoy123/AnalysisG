Particle Module (C++)
=====================

The Particle module provides the C++ implementation of particle templates.

Overview
--------

Located in ``src/AnalysisG/modules/particle/``, this module implements particle 
template functionality in C++:

- Particle kinematics (4-vector operations)
- Coordinate transformations (Cartesian ↔ Polar)
- PDG particle identification
- Parent-child relationships
- Particle properties and metadata

C++ Class: particle_template
-----------------------------

Header Location
~~~~~~~~~~~~~~~

``src/AnalysisG/modules/particle/include/templates/particle_template.h``

Constructors
~~~~~~~~~~~~

.. cpp:function:: particle_template()

   Default constructor.

.. cpp:function:: explicit particle_template(particle_t* p)

   Construct from particle data structure.

.. cpp:function:: explicit particle_template(particle_template* p, bool dump = false)

   Copy constructor with optional data dumping.

.. cpp:function:: explicit particle_template(double px, double py, double pz, double e)

   Construct from 4-vector components (Cartesian).

.. cpp:function:: explicit particle_template(double px, double py, double pz)

   Construct from 3-momentum components.

Kinematic Properties
~~~~~~~~~~~~~~~~~~~~

The ``cproperty`` template provides automatic setter/getter generation for kinematics:

**Polar Coordinates**

.. code-block:: cpp

   cproperty<double, particle_template> pt;   // Transverse momentum
   cproperty<double, particle_template> eta;  // Pseudorapidity
   cproperty<double, particle_template> phi;  // Azimuthal angle
   cproperty<double, particle_template> e;    // Energy

**Cartesian Coordinates**

.. code-block:: cpp

   cproperty<double, particle_template> px;   // x-momentum
   cproperty<double, particle_template> py;   // y-momentum
   cproperty<double, particle_template> pz;   // z-momentum

**Derived Properties**

.. code-block:: cpp

   cproperty<double, particle_template> mass;  // Invariant mass
   cproperty<double, particle_template> P;     // Total momentum magnitude
   cproperty<double, particle_template> beta;  // Velocity (v/c)

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: void to_cartesian()

   Convert from polar (pt, eta, phi) to Cartesian (px, py, pz) coordinates.
   Called automatically when Cartesian coordinates are accessed.

.. cpp:function:: void to_polar()

   Convert from Cartesian (px, py, pz) to polar (pt, eta, phi) coordinates.
   Called automatically when polar coordinates are accessed.

Particle Identification
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   cproperty<int, particle_template> pdgid;        // PDG particle ID
   cproperty<std::string, particle_template> symbol; // Particle symbol (e.g., "e", "mu")
   cproperty<double, particle_template> charge;    // Electric charge
   cproperty<std::string, particle_template> hash;  // Unique hash identifier

**Particle Type Checks**

.. cpp:function:: bool is(std::vector<int> p)

   Check if particle matches any of the given PDG IDs.

   :param p: Vector of PDG IDs to check against
   :return: True if particle matches any ID

.. code-block:: cpp

   cproperty<bool, particle_template> is_b;    // Is bottom quark/hadron
   cproperty<bool, particle_template> is_lep;  // Is lepton
   cproperty<bool, particle_template> is_nu;   // Is neutrino

Property Setters and Getters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each kinematic property has associated static setter/getter methods:

.. code-block:: cpp

   // Energy
   static void set_e(double*, particle_template*);
   static void get_e(double*, particle_template*);
   
   // Transverse momentum
   static void set_pt(double*, particle_template*);
   static void get_pt(double*, particle_template*);
   
   // Pseudorapidity
   static void set_eta(double*, particle_template*);
   static void get_eta(double*, particle_template*);
   
   // Azimuthal angle
   static void set_phi(double*, particle_template*);
   static void get_phi(double*, particle_template*);
   
   // Cartesian components
   static void set_px(double*, particle_template*);
   static void get_px(double*, particle_template*);
   static void set_py(double*, particle_template*);
   static void get_py(double*, particle_template*);
   static void set_pz(double*, particle_template*);
   static void get_pz(double*, particle_template*);
   
   // Mass
   static void set_mass(double*, particle_template*);
   static void get_mass(double*, particle_template*);
   
   // Derived properties (getters only)
   static void get_P(double*, particle_template*);
   static void get_beta(double*, particle_template*);
   
   // Particle ID
   static void set_pdgid(int*, particle_template*);
   static void get_pdgid(int*, particle_template*);
   static void set_symbol(std::string*, particle_template*);
   static void get_symbol(std::string*, particle_template*);
   static void set_charge(double*, particle_template*);
   static void get_charge(double*, particle_template*);
   
   // Hash (getter only)
   static void get_hash(std::string*, particle_template*);
   
   // Type checks (getters only)
   static void get_isb(bool*, particle_template*);
   static void get_islep(bool*, particle_template*);
   static void get_isnu(bool*, particle_template*);

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

The particle_template supports 4-vector arithmetic:

**Addition**

Particles can be added to combine their 4-vectors (useful for invariant mass calculations).

**Comparison**

Particles can be compared for equality based on their hash values.

Parent-Child Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~

Particles maintain decay chain information through parent-child relationships,
allowing reconstruction of decay topologies.

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/particle/cxx/particle_template.cxx`` - Main implementation
- ``src/AnalysisG/modules/particle/cxx/properties.cxx`` - Property implementations
- ``src/AnalysisG/modules/particle/cxx/kinematics.cxx`` - Kinematic calculations

**Python Binding**

- ``src/AnalysisG/core/particle_template.pyx`` - Cython wrapper
- ``src/AnalysisG/core/particle_template.pxd`` - Cython declarations

Usage from C++
--------------

.. code-block:: cpp

   #include <templates/particle_template.h>
   
   // Create particle with Cartesian coordinates
   particle_template* p1 = new particle_template(100.0, 50.0, 200.0, 250.0);
   
   // Create particle with polar coordinates
   particle_template* p2 = new particle_template();
   p2->pt = 150.0;
   p2->eta = 0.5;
   p2->phi = 1.2;
   p2->e = 200.0;
   
   // Automatic coordinate conversion
   double px = p2->px;  // Triggers to_cartesian()
   double pt = p1->pt;  // Triggers to_polar()
   
   // Set particle identification
   p1->pdgid = -11;  // Positron
   p1->charge = 1.0;
   
   // Check particle type
   bool is_electron = p1->is({11, -11});
   bool is_lepton = p1->is_lep;
   
   // Calculate derived properties
   double momentum = p1->P;
   double velocity = p1->beta;
   double mass = p1->mass;

Integration with Python
-----------------------

The C++ particle_template is wrapped in Python as ``ParticleTemplate``:

.. code-block:: python

   from AnalysisG.core.particle_template import ParticleTemplate
   
   class Jet(ParticleTemplate):
       def __init__(self):
           ParticleTemplate.__init__(self)
   
   # Create particle
   jet = Jet()
   jet.pt = 100.0
   jet.eta = 0.5
   jet.phi = 1.2
   jet.e = 150.0
   
   # Access Cartesian coordinates (automatically calculated)
   print(jet.px, jet.py, jet.pz)
   
   # Set PDG ID
   jet.pdgid = -11
   print(jet.symbol)  # "e"
   print(jet.is_lep)  # True

Coordinate System Details
--------------------------

**Polar to Cartesian Conversion**

.. math::

   p_x &= p_T \cos(\phi) \\
   p_y &= p_T \sin(\phi) \\
   p_z &= p_T \sinh(\eta)

**Cartesian to Polar Conversion**

.. math::

   p_T &= \sqrt{p_x^2 + p_y^2} \\
   \phi &= \arctan2(p_y, p_x) \\
   \eta &= \text{asinh}\left(\frac{p_z}{p_T}\right)

**Energy-Momentum Relations**

.. math::

   E^2 &= p_x^2 + p_y^2 + p_z^2 + m^2 \\
   m^2 &= E^2 - p_x^2 - p_y^2 - p_z^2 \\
   |\vec{p}| &= \sqrt{p_x^2 + p_y^2 + p_z^2} \\
   \beta &= \frac{|\vec{p}|}{E}

See Also
--------

* :doc:`../core/templates` - Python ParticleTemplate documentation
* :doc:`event` - Event template C++ implementation
* :doc:`../pyc/physics` - High-performance physics calculations
