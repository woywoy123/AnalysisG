.. _graph-template:

GraphTemplate Methods
---------------------

This part of the documentation highlights some useful features that are part of the template class.

.. cpp:class:: graph_template: public tools

   .. cpp:var::  event_t data

   .. cpp:var::  std::string filename

   .. cpp:var::  cproperty<long, graph_template> index

   .. cpp:var::  cproperty<std::string, graph_template> hash

   .. cpp:var::  cproperty<std::string, graph_template> tree 

   .. cpp:var::  cproperty<std::string, graph_template> name 

   .. cpp:function::  graph_template()

   .. cpp:function::  virtual ~graph_template()

   .. cpp:function::  virtual graph_template* clone()

   .. cpp:function::  virtual void CompileEvent()

   .. cpp:function::  graph_t* data_export()

   .. cpp:function::  void flush_particles()

   .. cpp:function::  graph_template* build(event_template* el)

   .. cpp:function::  void define_particle_nodes(std::vector<particle_template*>* prt)

   .. cpp:function::  void define_topology(std::function<bool(particle_template*, particle_template*)> fx)

   .. cpp:function::  bool operator == (graph_template& p)

   .. cpp:function::  template <typename G> \
                      G* get_event()

   .. cpp:function::  template <typename G, typename O, typename X> \
                      void add_graph_truth_feature(O* ev, X fx, std::string name)

   .. cpp:function::  template <typename G, typename O, typename X> \
                      void add_graph_data_feature(O* ev, X fx, std::string name)

   .. cpp:function::  template <typename G, typename O, typename X> \
                      void add_node_truth_feature(X fx, std::string name)

   .. cpp:function::  template <typename G, typename O, typename X> \
                      void add_node_data_feature(X fx, std::string name)

   .. cpp:function::  template <typename G, typename O, typename X> \
                      void add_edge_truth_feature(X fx, std::string name)

   .. cpp:function::  template <typename G, typename O, typename X> \
                      void add_edge_data_feature(X fx, std::string name)
