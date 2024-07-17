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
   graph_template::clone(){return (graph_template*)new graph_name();}

   void graph_name::CompileEvent(){
        //auto some_gfx(); 
        // get the event that has all the fill event data
        event_name* event = this -> get_event<event_name>(); 

        std::vector<particle_template*> particles = event -> some_var_with_particles; 
        
        // ----- truth features ----- //
        this -> add_graph_truth_feature<bool, event_name>(event, some_gfx, "name-this-truth");

   }



