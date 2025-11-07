graph_template.h
================

**File Path**: ``modules/graph/include/templates/graph_template.h``

**File Type**: H (Header)

**Lines**: 404

Dependencies
------------

**Includes**:

- ``c10/core/DeviceType.h``
- ``mutex``
- ``pyc/pyc.h``
- ``structs/enums.h``
- ``structs/event.h``
- ``structs/folds.h``
- ``structs/property.h``
- ``templates/event_template.h``
- ``templates/particle_template.h``
- ``tools/tensor_cast.h``
- ``tools/tools.h``
- ``torch/torch.h``

Classes
-------

``graph_template``
~~~~~~~~~~~~~~~~~~

**Inherits from**: ``tools``

**Methods**:

- ``void CompileEvent()``
- ``bool PreSelection()``
- ``void define_particle_nodes(std::vector<particle_template*>* prt)``
- ``void define_topology(std::function<bool(particle_template*, particle_te...)``
- ``void flush_particles()``
- ``void add_graph_truth_feature(O* ev, X fx, std::string _name)``
- ``void add_graph_data_feature(O* ev, X fx, std::string _name)``
- ``void add_node_truth_feature(X fx, std::string _name)``
- ``void add_node_data_feature(X fx, std::string _name)``
- ``void add_edge_truth_feature(X fx, std::string _name)``

Structs
-------

``graph_t``
~~~~~~~~~~~

**Members**:

- ``public: 
        template <typename g>
        torch::Tensor* get_truth_graph(std::string _name, g* mdl){
            return this -> has_feature(graph_enum::truth_graph, _name, mdl -> device_index)``

