The GraphTemplate Source Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the `event_template` class, the `graph_template` class allows for graphs to be constructed, that can be later used to train a Graph Neural Network.
In this class, specific types of particles can be used to construct `Nodes`, which are subsequently assigned some set of input features.
Assignment of features is performed by passing lambda functions that specifically retrieve such features.
For instance, given some callable function :math:`y_i = \mathcal{F}(x_i)`, with input particle :math:`x_i` and output feature `y_i`, the template class will assign each particle node the value :math:`y_i` in a graph.
By a similar token, to construct edge features, the callable function can be functionally expressed as :math:`y_{ij} = \mathcal{F}(x_i, x_j)`. 

To further simplify the pipeline, internal functions exploit dimensional analysis on the output vector space to infer the dimension of the `PyTorch` tensor and simply builds this tensor.

A simple code implementation would look structurally as below:

.. code:: C++

   #ifndef GRAPHS_GRAPHNAME_H
   #define GRAPHS_GRAPHNAME_H

   #include <event-name/some-event.h>

   class graph_name: public graph_template
   {
        public:
            graph_name();
            ~graph_name(); 
            
            graph_template* clone() override; // make the implemenation cloneable
            void CompileEvent() override; // Compiler space.

   }; 

   #endif

.. code:: C++

   #include "graph-name.h"

   // boiler plate code
   graph_name::graph_name(){this -> name = "graph-name";}
   graph_name::~graph_name(){}
   graph_template* graph_name::clone(){return (graph_template*)new graph_name();} // caller owns the returned pointer

   void graph_name::CompileEvent(){
        // Graph truth: is the event a signal?  ev->is_signal is a user-defined
        // attribute of event_name (bool or int declared in your event header).
        auto some_graph_fx = [](bool* out, event_name* ev){*out = ev -> is_signal;};

        // Node truth: is this particle a b-quark/hadron?
        // particle_template provides is_b, is_lep, is_nu, is_add directly.
        // For derived-class-specific attributes, cast first:
        //   your_type* t = (your_type*)p;  *out = t -> your_attr;
        auto some_node_fx = [](int* out, particle_template* p){*out = (int)(p -> is_b);};

        // Edge truth: do both endpoints share the same sequential index?
        // particle_template::index is a built-in cproperty<int> on every particle.
        auto some_edge_fx = [](int* out, std::tuple<particle_template*, particle_template*>* e_ij){
            *out = (std::get<0>(*e_ij) -> index == std::get<1>(*e_ij) -> index) ? 1 : 0;
        }; 

        // Graph observable: ev->met is a user-defined float/double in event_name.
        auto some_other_graph_fx = [](double* out, event_name* ev){*out = ev -> met;}; // get missing transverse momenta

        auto some_other_node_fx = [](double* out, particle_template* p){*out = p -> pt;}; // get particle pt

        auto some_other_edge_fx = [](double* out, std::tuple<particle_template*, particle_template*>* e_ij){
            *out = std::get<0>(*e_ij) -> pt - std::get<1>(*e_ij) -> pt; // pt difference between particles
        }; 

    
        // get the event that has all the fill event data
        event_name* event = this -> get_event<event_name>(); 
        std::vector<particle_template*> particles = event -> some_var_with_particles; 
       
        // Register particle nodes first — must be called before any feature functions.
        this -> define_particle_nodes(&particles);

        // ------ define a prior topology or leave this out for a fully connected graph ----- //
        // for instance, if the target particle is a top, then no two b-jets can share the same top.
        auto bias_topology = [](particle_template* p1, particle_template* p2){return p1 -> is_b != p2 -> is_b;};
        this -> define_topology(bias_topology);  // this topology bias will be reflected in subsequent topology functions.

        // ----- truth features ----- //
        this -> add_graph_truth_feature<bool, event_name>(event, some_graph_fx, "name-this-truth");
        this -> add_node_truth_feature<int, particle_template>(some_node_fx, "is_top");
        this -> add_edge_truth_feature<int, particle_template>(some_edge_fx, "top_edge"); 

        // ----- observables --------- //
        this -> add_graph_data_feature<double, event_name>(event, some_other_graph_fx, "some-other-data"); 
        this -> add_node_data_feature<double, particle_template>(some_other_node_fx, "pt"); 
        this -> add_edge_data_feature<double, particle_template>(some_other_edge_fx, "delta-pt"); 

        // The class also allows for access to event data that relates to the origin root file.
        std::string root_file_path = this -> filename; 
        
        // or event_t struct access
        event_t ev_data = this -> data; 
   }


The above code is all that is required to construct graphs for Graph Neural Network training! 
There are no additional steps required to start using the `PyTorch` API.

For more information about the methods and attributes of the `graph_template` class, see the core-class documentation :ref:`graph-template`.

