.. _graph-template:

GraphTemplate Methods
---------------------

This part of the documentation highlights some useful features that are part of the template class.
To use the class, import the following header 

.. code:: C++

   #include <templates/graph_template.h>

.. cpp:class:: graph_template: public tools

   .. cpp:var::  event_t data

      A struct holding event information.

   .. cpp:var::  std::string filename

      A variable holding the absolute path of the origin ROOT file.

   .. cpp:var::  cproperty<long, graph_template> index

      A variable used to set and retrieve the current event index. 
      It value is sourced from the `data` attribute.

   .. cpp:var::  cproperty<std::string, graph_template> hash

      Holds a non-cryptographically unique identifier used to mark the current event.

   .. cpp:var::  cproperty<std::string, graph_template> tree 
     
      The current event tree from which the input event_template is sourced from.

   .. cpp:var::  cproperty<std::string, graph_template> name 

      A varible used to assign the graph_template a name.

   .. cpp:function:: graph_template()

      A default constructor.

   .. cpp:function:: virtual ~graph_template()

      A default object destructor.

   .. cpp:function:: virtual graph_template* clone()

      A function used to clone the current graph_template implementation.
      It is important to override this function, since it allows the framework to make multiple duplicates of the graph_template.
      To use it, simply override the function in the header file, and define a function with contents; `return new <graph name>();`.

   .. cpp:function:: virtual void CompileEvent()

      A function to be overriden for custom graph generation. 
      This function is used to build the tensors used in the model training/inference.

   .. cpp:function:: graph_t* data_export()

      Exports the current graph tensors.
      For general usage, this function should not be called.

   .. cpp:function:: void flush_particles()

      Purges all particles from the graph.

   .. cpp:function:: graph_template* build(event_template* el)

      A function used to build the graph, this function should not be overriden.

   .. cpp:function:: void define_particle_nodes(std::vector<particle_template*>* prt)

      Defines the particles used as nodes in a graph.

   .. cpp:function:: void define_topology(std::function<bool(particle_template*, particle_template*)> fx)

      A function which takes as input a function used to define any prior topology construction of a graph.
      If no prior knowledge of the graph connectivity is known, call the function with the `fulltopo` function, which creates a fully connected graph.

   .. cpp:function:: bool operator == (graph_template& p)

   .. cpp:function:: template <typename G> \
                     G* get_event()

      Fetches the event_template implementation.

   .. cpp:function:: template <typename G, typename O, typename X> \
                     void add_graph_truth_feature(O* ev, X fx, std::string name)

      A function used to define truth information at the graph level.
      To use it, simply pass the event pointer, the function, and the desired attribute name.

   .. cpp:function:: template <typename G, typename O, typename X> \
                     void add_graph_data_feature(O* ev, X fx, std::string name)

      A function used to define data information at the graph level.
      To use it, simply pass the event pointer, the function, and the desired attribute name.

   .. cpp:function:: template <typename G, typename O, typename X> \
                     void add_node_truth_feature(X fx, std::string name)

      A function used to define truth information at the node level.
      To use it, simply pass the function, and the desired attribute name.

   .. cpp:function:: template <typename G, typename O, typename X> \
                     void add_node_data_feature(X fx, std::string name)

      A function used to define data information at the node level.
      To use it, simply pass the function, and the desired attribute name.

   .. cpp:function:: template <typename G, typename O, typename X> \
                     void add_edge_truth_feature(X fx, std::string name)

      A function used to define truth information at the edge level.
      To use it, simply pass the function, and the desired attribute name. 

   .. cpp:function:: template <typename G, typename O, typename X> \
                     void add_edge_data_feature(X fx, std::string name)

      A function used to define data information at the edge level.
      To use it, simply pass the function, and the desired attribute name. 

graph_t Methods
---------------

A struct which is derived from the `graph_template`, it holds all tensors and handles cross device transfers, such that multiple models can be trained on different GPU devices in a single session.
A full list of its public functions is provided below.

.. cpp:struct:: graph_t

   .. cpp:function:: template <typename g> torch::Tensor* get_truth_graph(std::string, g* model); 
      
      A function which returns a tensor pointer relating to truth graph information, as defined by the graph_template.
      If the requsted attribute has not been found, a null pointer is returned.

   .. cpp:function:: template <typename g> torch::Tensor* get_truth_node(std::string, g* model); 
      
      A function which returns a tensor pointer relating to truth node information, as defined by the graph_template.
      If the requsted attribute has not been found, a null pointer is returned.

   .. cpp:function:: template <typename g> torch::Tensor* get_truth_edge(std::string, g* model); 
      
      A function which returns a tensor pointer relating to truth edge information, as defined by the graph_template.
      If the requsted attribute has not been found, a null pointer is returned.


   .. cpp:function:: template <typename g> torch::Tensor* get_data_graph(std::string, g* model); 
      
      A function which returns a tensor pointer relating to data graph information, as defined by the graph_template.
      If the requsted attribute has not been found, a null pointer is returned.

   .. cpp:function:: template <typename g> torch::Tensor* get_data_node(std::string, g* model); 
      
      A function which returns a tensor pointer relating to data node information, as defined by the graph_template.
      If the requsted attribute has not been found, a null pointer is returned.

   .. cpp:function:: template <typename g> torch::Tensor* get_data_edge(std::string, g* model); 
      
      A function which returns a tensor pointer relating to data edge information, as defined by the graph_template.
      If the requsted attribute has not been found, a null pointer is returned.

   .. cpp:function:: template <typename g> torch::Tensor* get_edge_index(g* model); 
      
      Returns a pointer relating to the edge index, as defined by the event topology in the graph_template.
      It is used to index event nodes to edge features and returns a null pointer if not defined.

  .. cpp:var:: int num_nodes
  
     A variable used to indicate the number of nodes in the current graph.

  .. cpp:var:: long event_index

     A variable indicating the current event index from which the graph has been compiled from.

  .. cpp:var:: std::string* hash

     A pointer which relates to the non-cryptographically generated unique identifier of the event.

  .. cpp:var:: std::string* filename

     A pointer referencing the filename the graph has been generated from.

  .. cpp:var:: std::string* graph_name 
   
     A pointer referencing the graph_template used to generate the current graph_t data.


Extending Suppored Tensor Data Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since tensors can have multiple dimensions with a variety of data types, the current collection may not be sufficient for certain needs.
As such, the framework has been designed to easily extendable beyond the current data type collections.
Simply navigate to the source directory of the graph_template (src/AnalysisG/modules/graph/include/templates/graph_template.h), and add a new function to the private section of the `graph_template` class.
For instance, if a nested vector of its is required for training a model, simply follow the prescription below;

.. code:: C++ 

   // header declaration
   #include <templates/graph_template.h>

   class graph_template: public tools
   {
        private:

        //... 

        void add_node_feature(std::vector<std::vector<std::vector<int>>>, std::string); 

        //...
   }; 

   // source file (src/AnalysisG/modules/graph/cxx/properties.cxx)
   void graph_template::add_node_feature(std::vector<std::vector<std::vector<int>>> data, std::string name){
        this -> node_fx[name] = this -> to_tensor(data, torch::kInt, int()); 
   }


After adding this function, recompile the framework, and the new data type should be available for use.

