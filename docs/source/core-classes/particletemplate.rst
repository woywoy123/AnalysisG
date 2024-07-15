.. _particle-template:

ParticleTemplate Methods
------------------------

This part of the documentation highlights some useful features that are part of the template class. 
The particle_template class inherits from the tools class and uses the particle_t struct defined under the structs module.

.. cpp:class:: particle_template: public tools

   .. cpp:function:: particle_template()

   An empty constructor.

   .. cpp:function:: particle_template(particle_t* p)

   Create a new particle from an existing one.

   .. cpp:function:: particle_template(double px, double py, double pz, double e)

   Assign the four-momenta of the particle.

   .. cpp:function:: particle_template(double px, double py, double pz)

   Assing the three-momenta of the particle.

   .. cpp:var:: cproperty<double, particle_template> e

   The energy of the particle.

   .. cpp:var:: cproperty<double, particle_template> mass

   The invariant mass either computed from the particle's 4-vector or as given by the n-tuple.

   .. cpp:var:: cproperty<double, particle_template> pt

   Returns or sets the transverse momenta of the particle.

   .. cpp:var:: cproperty<double, particle_template> eta

   Returns or sets the pseudo-rapidity of the particle.

   .. cpp:var:: cproperty<double, particle_template> phi

   Returns or sets the azimuthal angle of the particle.

   .. cpp:var:: cproperty<double, particle_template> px

   Returns or sets the x-direction of the momenta.

   .. cpp:var:: cproperty<double, particle_template> py

   Returns or sets the y-direction of the momenta.

   .. cpp:var:: cproperty<double, particle_template> pz

   Returns or sets the z-direction of the momenta.

   .. cpp:var:: cproperty<int, particle_template> pdgid

   Returns or sets the particle PDGID.

   .. cpp:var:: cproperty<std::string, particle_template> symbol

   Returns the PDGID symbolic string representation or sets its value.

   .. cpp:var:: cproperty<double, particle_template> charge

   Assigns the particle a charge or returns its value.

   .. cpp:var:: cproperty<std::string, particle_template> hash

   Returns the hash of the particle (a unique identifier) by concatinating the cartesian 
   four-momenta and computes the associated hash.

   .. cpp:var:: cproperty<bool, particle_template> is_lep

   Returns whether the particle is a lepton.

   .. cpp:var:: cproperty<bool, particle_template> is_nu

   Returns whether the particle is a neutrino.

   .. cpp:var:: cproperty<bool, particle_template> is_add

   Returns whether the particle is anything but a b-quark/jet.

   .. cpp:var:: cproperty<bool, particle_template> is_b

   Returns whether the particle is a b-quark/jet.

   .. cpp:var:: cproperty<bool, particle_template> lep_decay

   Returns a boolean value indicating whether the decay was leptonic from its children.

   .. cpp:var:: cproperty<std::map<std::string, particle_template*>, particle_template> parents

   Returns the parents of the particle.

   .. cpp:var:: cproperty<std::map<std::string, particle_template*>, particle_template> children

   Returns the particle's children.

   .. cpp:var:: cproperty<std::string, particle_template> type

   Specifies the particle type.

   .. cpp:var:: cproperty<int, particle_template> index

   Assigns the particle an index.

   .. cpp:var:: std::map<std::string, particle_template*> m_parents

   A map of the particle's parent hashes.

   .. cpp:var:: std::map<std::string, particle_template*> m_children

   A map of the particle's children hashes.

   .. cpp:var:: std::map<std::string, std::string> leaves

   .. cpp:var:: particle_t data

   .. cpp:function:: void to_cartesian()

   .. cpp:function:: void to_polar()

   .. cpp:function:: void static set_e(double*, particle_template*)

   .. cpp:function:: void static get_e(double*, particle_template*)

   .. cpp:function:: void static set_mass(double*, particle_template*)

   .. cpp:function:: void static get_mass(double*, particle_template*)

   .. cpp:function:: void static set_pt(double*, particle_template*)

   .. cpp:function:: void static get_pt(double*, particle_template*)

   .. cpp:function:: void static set_eta(double*, particle_template*)

   .. cpp:function:: void static get_eta(double*, particle_template*)

   .. cpp:function:: void static set_phi(double*, particle_template*)

   .. cpp:function:: void static get_phi(double*, particle_template*)

   .. cpp:function:: void static set_px(double*, particle_template*)

   .. cpp:function:: void static get_px(double*, particle_template*)

   .. cpp:function:: void static set_py(double*, particle_template*)

   .. cpp:function:: void static get_py(double*, particle_template*)

   .. cpp:function:: void static set_pz(double*, particle_template*)

   .. cpp:function:: void static get_pz(double*, particle_template*)

   .. cpp:function:: void static set_pdgid(int*, particle_template*)

   .. cpp:function:: void static get_pdgid(int*, particle_template*)

   .. cpp:function:: void static set_symbol(std::string*, particle_template*)

   .. cpp:function:: void static get_symbol(std::string*, particle_template*)

   .. cpp:function:: void static set_charge(double*, particle_template*)

   .. cpp:function:: void static get_charge(double*, particle_template*)

   .. cpp:function:: void static get_hash(std::string*, particle_template*)

   .. cpp:function:: bool is(std::vector<int> p)

   .. cpp:function:: void static get_isb(bool*, particle_template*)

   .. cpp:function:: void static get_islep(bool*, particle_template*)

   .. cpp:function:: void static get_isnu(bool*, particle_template*)

   .. cpp:function:: void static get_isadd(bool*, particle_template*)

   .. cpp:function:: void static get_lepdecay(bool*, particle_template*)

   .. cpp:function:: void static set_parents(std::map<std::string, particle_template*>*, particle_template*)

   .. cpp:function:: void static get_parents(std::map<std::string, particle_template*>*, particle_template*)

   .. cpp:function:: void static set_children(std::map<std::string, particle_template*>*, particle_template*)

   .. cpp:function:: void static get_children(std::map<std::string, particle_template*>*, particle_template*)

   .. cpp:function:: void static set_type(std::string*, particle_template*)

   .. cpp:function:: void static get_type(std::string*, particle_template*)

   .. cpp:function:: void static set_index(int*, particle_template*)

   .. cpp:function:: void static get_index(int*, particle_template*)

   .. cpp:function:: double DeltaR(particle_template* p)

   .. cpp:function:: bool operator == (particle_template& p)

   .. cpp:function:: template <typename g> \
                     g operator + (g& p)

   .. cpp:function:: void operator += (particle_template* p)

   .. cpp:function:: void iadd(particle_template* p)

   .. cpp:function:: bool register_parent(particle_template* p)

   .. cpp:function:: bool register_child(particle_template* p)

   .. cpp:function:: void add_leaf(std::string key, std::string leaf)

   .. cpp:function:: void apply_type_prefix()

   .. cpp:function:: virtual void build(std::map<std::string, particle_template*>* event, element_t* el)

   .. cpp:function:: virtual particle_template* clone()


  
