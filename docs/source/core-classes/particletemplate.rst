ParticleTemplate Methods
------------------------

.. cpp:class:: particle_template: public tools

   .. cpp:var:: cproperty<double, particle_template> e

   The energy of the particle.

   .. cpp:var:: cproperty<double, particle_template> mass

   The invariant mass either computed from the particle's 4-vector or as given by the n-tuple.

   .. cpp:var:: cproperty<double, particle_template> pt

   .. cpp:var:: cproperty<double, particle_template> eta

   .. cpp:var:: cproperty<double, particle_template> phi

   .. cpp:var:: cproperty<double, particle_template> px

   .. cpp:var:: cproperty<double, particle_template> py

   .. cpp:var:: cproperty<double, particle_template> pz

   .. cpp:var:: cproperty<int, particle_template> pdgid

   .. cpp:var:: cproperty<std::string, particle_template> symbol

   .. cpp:var:: cproperty<double, particle_template> charge

   .. cpp:var:: cproperty<std::string, particle_template> hash

   .. cpp:var:: cproperty<bool, particle_template> is_lep

   .. cpp:var:: cproperty<bool, particle_template> is_nu

   .. cpp:var:: cproperty<bool, particle_template> is_add

   .. cpp:var:: cproperty<bool, particle_template> lep_decay

   .. cpp:var:: cproperty<bool, particle_template> is_b

   .. cpp:var:: cproperty<std::map<std::string, particle_template*>, particle_template> parents

   .. cpp:var:: cproperty<std::map<std::string, particle_template*>, particle_template> children

   .. cpp:var:: cproperty<std::string, particle_template> type

   .. cpp:var:: cproperty<int, particle_template> index

   .. cpp:var:: std::map<std::string, particle_template*> m_parents

   .. cpp:var:: std::map<std::string, particle_template*> m_children

   .. cpp:var:: std::map<std::string, std::string> leaves

   .. cpp:var:: particle_t data

   .. cpp:function:: particle_template(particle_t* p)

   .. cpp:function:: particle_template(double px, double py, double pz, double e)

   .. cpp:function:: particle_template(double px, double py, double pz)

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


  
