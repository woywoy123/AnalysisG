.. _tutorial-custom-graph:

Tutorial: Creating and Integrating a Custom C++ Graph Template
=============================================================

This tutorial guides you through creating a custom C++ graph definition within the AnalysisG framework, linking it using CMake, and making it accessible from Python via Cython.

We'll create a simple graph definition called ``MyCustomGraph``.

1. Implement the C++ Graph Template
-----------------------------------

First, define your custom graph class in C++. This involves creating a header (``.h``) and a source (``.cxx`` or ``.cpp``) file.

**Header File (``include/MyCustomGraph.h``)**

Create a header file for your new graph class. It should inherit from ``graph_template``.

.. code-block:: cpp

    #ifndef ANALYSISG_MYCUSTOMGRAPH_H
    #define ANALYSISG_MYCUSTOMGRAPH_H

    #include "Event/event_template.h" // Or your specific event header
    #include "Graph/graph_template.h"

    namespace AnalysisG
    {
        class MyCustomGraph : public graph_template
        {
            public:
                // Constructor: Sets the unique name
                MyCustomGraph();

                // Destructor: Cleans up resources if needed
                ~MyCustomGraph() override;

                // clone: Required virtual method for copying
                graph_template* clone() override;

                // CompileEvent: Defines how to build the graph from an event
                void CompileEvent() override;
        };
    }

    #endif // ANALYSISG_MYCUSTOMGRAPH_H

**Source File (``src/Graph/MyCustomGraph.cxx``)**

Implement the methods declared in the header file.

.. code-block:: cpp

    #include "MyCustomGraph.h"
    #include "Event/event_template.h" // Or your specific event header
    #include "Particles/particle_template.h" // Or your specific particle header

    namespace AnalysisG
    {
        // --- Constructor ---
        // Set the unique name for this graph definition.
        // This name will be used to select it in Python/configuration.
        MyCustomGraph::MyCustomGraph() { this->name = "MyCustomGraphName"; }

        // --- Destructor ---
        // Usually empty unless specific cleanup is needed.
        MyCustomGraph::~MyCustomGraph() {}

        // --- Clone Method ---
        // Returns a new instance of this graph class.
        graph_template* MyCustomGraph::clone()
        {
            return static_cast<graph_template*>(new MyCustomGraph());
        }

        // --- CompileEvent Method ---
        // This is where you define the graph structure and features.
        void MyCustomGraph::CompileEvent()
        {
            // 1. Get the event object. Replace 'event_template' if using a specific event type.
            event_template* event = this->get_event<event_template>();

            // 2. Select particles to become nodes.
            // Example: Use all particles from a hypothetical 'particles' collection.
            // Replace 'event->particles' with your actual particle collection.
            std::vector<particle_template*> nodes = event->particles;
            this->set_nodes(nodes); // Assign these particles as graph nodes

            // 3. Define Node Features.
            // Example: Add particle transverse momentum (pt).
            auto node_pt_func = [](std::vector<double>* out, particle_template* p) {
                out->push_back(p->pt);
            };
            this->add_node_data_feature<double>(node_pt_func, "node_pt");

            // Example: Add particle eta.
            auto node_eta_func = [](std::vector<double>* out, particle_template* p) {
                out->push_back(p->eta);
            };
            this->add_node_data_feature<double>(node_eta_func, "node_eta");

            // 4. Define Edge Features (assuming a fully connected graph for simplicity).
            // Example: Add the difference in pt between connected nodes.
            auto edge_dpt_func = [](std::vector<double>* out, particle_template* p1, particle_template* p2) {
                out->push_back(p1->pt - p2->pt);
            };
            this->add_edge_data_feature<double>(edge_dpt_func, "edge_delta_pt");

            // Example: Add the geometric distance (Delta R) between nodes.
            auto edge_dr_func = [](std::vector<double>* out, particle_template* p1, particle_template* p2) {
                out->push_back(p1->DeltaR(p2)); // Assuming particle_template has a DeltaR method
            };
            this->add_edge_data_feature<double>(edge_dr_func, "edge_delta_r");

            // Note: Topology (how nodes are connected) might be set elsewhere
            // or implicitly assumed (e.g., fully connected if not specified).
            // Refer to AnalysisG documentation for topology definition.
        }
    } // namespace AnalysisG

2. Link with CMake
------------------

Now, tell CMake about your new C++ files so they get compiled and linked into the AnalysisG library.

*   Locate the main ``CMakeLists.txt`` file for the AnalysisG framework (usually in the project root or a ``src`` directory).
*   Find the section where source files are added to the core AnalysisG library target (e.g., using ``target_sources`` or adding to a list variable like ``ANALYSISG_SOURCES``).
*   Add your new source file (``src/Graph/MyCustomGraph.cxx``) to this list.

**Example ``CMakeLists.txt`` Snippet:**

.. code-block:: cmake

    # ... other CMake commands ...

    # List of source files for the main library
    set(ANALYSISG_SOURCES
        src/Graph/graph_template.cxx
        src/Event/event_template.cxx
        # ... other existing source files ...
        src/Graph/MyCustomGraph.cxx   # <--- Add your new source file here
    )

    # Add sources to the library target (target name might differ)
    target_sources(AnalysisG PRIVATE ${ANALYSISG_SOURCES})

    # ... other CMake commands ...

*   After modifying ``CMakeLists.txt``, re-run the CMake configuration step and build the project:

    .. code-block:: bash

        cd /path/to/your/build/directory
        cmake /path/to/AnalysisG/source
        make # or ninja, depending on your generator

3. Expose to Python with Cython
-------------------------------

AnalysisG uses Cython to create Python bindings for its C++ code. You need to make Cython aware of your new ``MyCustomGraph`` class.

*   **Declaration (``.pxd`` file):** You might need to declare the C++ class in a relevant Cython declaration file (e.g., ``AnalysisG/bindings/graph.pxd``). This tells Cython about the C++ class structure.

    .. code-block:: cython

        # In a relevant .pxd file (e.g., AnalysisG/bindings/graph.pxd)
        cdef extern from "MyCustomGraph.h" namespace "AnalysisG":
            cdef cppclass MyCustomGraph(graph_template):
                MyCustomGraph() except +
                # Declare other methods if they need to be called directly from Cython

*   **Factory/Registry:** AnalysisG likely uses a factory pattern or registry to manage different graph template types. You need to ensure your ``MyCustomGraph`` is registered so it can be instantiated based on its name (``"MyCustomGraphName"``). This usually involves:
    *   Including the C++ header (``MyCustomGraph.h``) in the C++ file where the factory/registry is implemented.
    *   Adding an instance or a creation function for ``MyCustomGraph`` to the factory's map or list, associated with its name.
    *   *Look for existing examples within the AnalysisG codebase (e.g., how ``graph_sum_feature`` or other standard graphs are registered) and follow that pattern.* The exact mechanism is framework-specific.

*   **Rebuild:** After modifying Cython (``.pxd``, ``.pyx``) or C++ factory files, rebuild the project (including the Python extension modules).

    .. code-block:: bash

        cd /path/to/AnalysisG/source
        pip install -e . # Or the framework's specific build command

4. Use in Python
----------------

Once compiled and the Cython bindings are updated, you should be able to select and use your custom graph template in Python scripts by referring to the name you assigned in the C++ constructor (``"MyCustomGraphName"``).

.. code-block:: python

    import AnalysisG

    # Assuming AnalysisG uses a configuration dictionary or similar mechanism
    # The exact usage depends on the AnalysisG analysis setup script.
    config = {
        # ... other configurations ...
        "graph_name": "MyCustomGraphName", # Select your custom graph
        # ... other configurations ...
    }

    # Example: Running an analysis (replace with actual AnalysisG workflow)
    # analysis = AnalysisG.Analysis(config)
    # analysis.run()

    # Or if instantiating directly (less common for full analysis)
    # try:
    #     graph = AnalysisG.Graph("MyCustomGraphName")
    #     print(f"Successfully created graph: {graph.name}")
    # except Exception as e:
    #     print(f"Failed to create graph: {e}")

    print(f"Using custom graph template: {config['graph_name']}")
    # Proceed with your AnalysisG workflow...

This provides a basic structure. You'll need to adapt the file paths, class names (``event_template``, ``particle_template``), CMake target names, and Cython integration details based on the specific structure and conventions of the AnalysisG framework you are working with. Always refer to existing code within AnalysisG as the primary guide.
