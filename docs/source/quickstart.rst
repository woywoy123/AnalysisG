===========
Quick Start
===========

This guide will help you get started with AnalysisG by walking through a simple example.

Basic Concepts
==============

AnalysisG uses a template-based design where you:

1. **Define Events**: Specify how to read your data
2. **Define Particles**: Specify particle properties
3. **Define Graphs**: Specify how events become graphs
4. **Define Models**: Specify your neural network
5. **Run Analysis**: Let the framework handle the rest

A Simple Example
================

Here's a complete example of setting up an analysis:

Step 1: Define Your Event
-------------------------

.. code-block:: cpp

    #include <templates/event_template.h>

    class MyEvent : public event_template {
    public:
        std::vector<double> jet_pt;
        std::vector<double> jet_eta;
        double met;
        
        MyEvent() {
            add_tree("nominal");
            add_leaf("jet_pt");
            add_leaf("jet_eta");
            add_leaf("met");
        }
        
        void build(element_t* el) override {
            // Build logic here
        }
        
        MyEvent* clone() override {
            return new MyEvent();
        }
    };

Step 2: Define Your Graph
-------------------------

.. code-block:: cpp

    #include <templates/graph_template.h>

    class MyGraph : public graph_template {
    public:
        MyGraph() {
            name = "my_graph";
        }
        
        void CompileEvent() override {
            MyEvent* ev = get_event<MyEvent>();
            
            // Add node features
            add_node_data_feature<double, particle_template>(
                [](particle_template* p) { return p->pt(); }, 
                "pt"
            );
            
            // Add graph features
            add_graph_data_feature<double, MyEvent>(
                ev, 
                [](MyEvent* e) { return e->met; }, 
                "met"
            );
        }
    };

Step 3: Run the Analysis
------------------------

.. code-block:: cpp

    #include <AnalysisG/analysis.h>

    int main() {
        analysis ana;
        
        // Configure settings
        ana.m_settings.output = "output/";
        ana.m_settings.threads = 4;
        
        // Add samples
        ana.add_samples("/path/to/data/", "sample_label");
        
        // Add templates
        ana.add_event_template(new MyEvent(), "sample_label");
        ana.add_graph_template(new MyGraph(), "sample_label");
        
        // Start processing
        ana.start();
        
        return 0;
    }

Python Interface
================

AnalysisG also provides a Python interface:

.. code-block:: python

    from AnalysisG import analysis
    from AnalysisG.events import MyEvent
    from AnalysisG.graphs import MyGraph

    # Create analysis
    ana = analysis()
    
    # Configure
    ana.output = "output/"
    ana.threads = 4
    
    # Add data and templates
    ana.add_samples("/path/to/data/", "sample")
    ana.add_event_template(MyEvent(), "sample")
    ana.add_graph_template(MyGraph(), "sample")
    
    # Run
    ana.start()

Next Steps
==========

- Read the :doc:`tutorials` for more detailed examples
- Check the :doc:`api/library_root` for API reference
- See :doc:`contributing` if you want to contribute
