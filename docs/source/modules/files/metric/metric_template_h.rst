metric_template.h
=================

**File Path**: ``modules/metric/include/templates/metric_template.h``

**File Type**: H (Header)

**Lines**: 218

Dependencies
------------

**Includes**:

- ``meta/meta.h``
- ``notification/notification.h``
- ``plotting/plotting.h``
- ``structs/element.h``
- ``structs/enums.h``
- ``structs/event.h``
- ``structs/model.h``
- ``structs/property.h``
- ``templates/particle_template.h``
- ``tools/merge_cast.h``
- ``tools/tools.h``
- ``tools/vector_cast.h``

Classes
-------

``metric_template``
~~~~~~~~~~~~~~~~~~~

**Inherits from**: ``tools, 
    public notification``

**Methods**:

- ``void define_variables()``
- ``void define_metric(metric_t* v)``
- ``void event()``
- ``void batch()``
- ``void end()``
- ``void register_output(std::string tree, std::string __name, T* t)``
- ``new writer()``
- ``void write(std::string tree, std::string __name, T* t, bool f...)``
- ``void sum(std::vector<g*>* ch, k** out)``
- ``new k()``

Structs
-------

``graph_t``
~~~~~~~~~~~

``metric_t``
~~~~~~~~~~~~

**Members**:

- ``public: 
        ~metric_t()``
- ``int kfold = 0``
- ``int epoch = 0``
- ``int device = 0``
- ``template <typename g>
        g get(graph_enum grx, std::string _name){
            g out = g()``
- ``if (!this -> h_maps[grx][_name]){
                std::cout << "\033[1``
- ``31m Variable not found: " << _name << "\033[0m" << std::endl``
- ``return out``

