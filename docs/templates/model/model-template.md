# Model Template

.. _model-template-guide:

Adding a New Model From Template
================================

This guide walks through creating a new model for the `AnalysisG` framework using the provided templates.

**1. Copy and Rename Template Files**

*   Navigate to the model template directory: ``src/AnalysisG/templates/model``.
*   Copy this entire ``model`` directory into the main models directory: ``src/AnalysisG/models/``.
*   Rename the newly copied directory from ``model`` to your desired model name, e.g., ``src/AnalysisG/models/MyNewModel``.
*   Inside your new model directory (``src/AnalysisG/models/MyNewModel``), rename the template files:
    *   ``include/models/model_template.h`` -> ``include/models/MyNewModel.h``
    *   ``cxx/model_template.cxx`` -> ``cxx/MyNewModel.cxx``
    *   ``cython/model_template.pxd`` -> ``cython/MyNewModel.pxd``
    *   ``cython/model_template.pyx`` -> ``cython/MyNewModel.pyx``
    *   ``CMakeLists.txt`` remains ``CMakeLists.txt``.

**2. Modify C++ Source Code**

Replace all occurrences of ``<model-name>`` with your C++ class name (e.g., ``MyNewModel``).

*   **Header File (``include/models/MyNewModel.h``)**: Defines the C++ model class structure.

    .. code-block:: c++
        :caption: include/models/MyNewModel.h
        :linenos:

        #ifndef MYNEWMODEL_H // <- Change guard
        #define MYNEWMODEL_H // <- Change guard
        #include <templates/model_template.h>

        class MyNewModel: public model_template // <- Change class name
        {
            public:
                MyNewModel(); // <- Change constructor name
                ~MyNewModel(); // <- Change destructor name
                model_template* clone() override;
                void forward(graph_t*) override;

                // Add specific member variables for your model here
                torch::nn::Sequential* example = nullptr;
        };

        #endif

*   **Source File (``cxx/MyNewModel.cxx``)**: Implements the C++ model class methods.

    .. code-block:: c++
        :caption: cxx/MyNewModel.cxx
        :linenos:

        #include <MyNewModel.h> // <- Change include

        // Constructor: Initialize layers, register modules
        MyNewModel::MyNewModel(){ // <- Change class::constructor name

            this -> example = new torch::nn::Sequential({
                    {"L1", torch::nn::Linear(2, 2)},
                    {"RELU", torch::nn::ReLU()},
                    {"L2", torch::nn::Linear(2, 2)}
            });
            // Register your model components
            this -> register_module("example_sequential", this -> example);
        }

        // Destructor: Clean up if needed (often empty)
        MyNewModel::~MyNewModel(){} // <- Change class::destructor name

        // Clone method: Used internally by the framework
        model_template* MyNewModel::clone(){ // <- Change class name
            return new MyNewModel(); // <- Change class name
        }

        // Forward method: Define your model's computation logic
        void MyNewModel::forward(graph_t* data){ // <- Change class name

            // Fetch input data (example)
            torch::Tensor node_features = data->get_data_node("features")->clone();
            // ... fetch other needed data ...

            // --- Implement your model logic here ---
            // Example: Pass data through the sequential layer
            torch::Tensor output = this->example->forward(node_features);
            // --- End of model logic ---

            // Store prediction outputs
            this -> prediction_node_feature("output_prediction", output);
            // ... store other predictions (graph, edge, extra) ...
        }

**3. Modify Cython Interface Files**

Replace ``<model-name>`` with your C++ class name (e.g., ``MyNewModel``) and ``<py-model-name>`` with your desired Python class name (e.g., ``PyMyNewModel``).

*   **Cython Header (``cython/MyNewModel.pxd``)**: Declares the C++ class to Cython.

    .. code-block:: cython
        :caption: cython/MyNewModel.pxd
        :linenos:

        # distutils: language=c++
        # cython: language_level=3

        from libcpp cimport bool
        from AnalysisG.core.model_template cimport model_template, ModelTemplate

        # Point to your C++ header file
        cdef extern from "<models/MyNewModel.h>":
            # Declare your C++ class inheriting from model_template
            cdef cppclass MyNewModel(model_template):
                MyNewModel() except+ # Constructor
                # Declare any C++ members you want accessible (optional)
                # bool example_flag

        # Define the Python wrapper class inheriting from ModelTemplate
        cdef class PyMyNewModel(ModelTemplate):
            pass

*   **Cython Source (``cython/MyNewModel.pyx``)**: Implements the Python wrapper class.

    .. code-block:: cython
        :caption: cython/MyNewModel.pyx
        :linenos:

        # distutils: language=c++
        # cython: language_level=3

        from AnalysisG.core.model_template cimport ModelTemplate
        # Import the C++ class and Python wrapper from the .pxd file
        from AnalysisG.models.MyNewModel.MyNewModel cimport MyNewModel, PyMyNewModel

        # Implement the Python wrapper class
        cdef class PyMyNewModel(ModelTemplate):
            def __cinit__(self):
                # Create an instance of your C++ class
                self.nn_ptr = new MyNewModel()
            def __init__(self):
                # Standard Python initializer (usually pass)
                pass
            def __dealloc__(self):
                # Clean up the C++ instance
                del self.nn_ptr

            # Add Python methods/properties to interact with C++ members if needed
            # property example_flag:
            #     def __get__(self): return (<MyNewModel*>self.nn_ptr).example_flag
            #     def __set__(self, bool value): (<MyNewModel*>self.nn_ptr).example_flag = value

**4. Modify CMake Build File**

Update the ``CMakeLists.txt`` inside your model directory (``src/AnalysisG/models/MyNewModel/CMakeLists.txt``). Replace ``<model-name>`` with your C++ class name (e.g., ``MyNewModel``) and adjust dependencies if needed.

.. code-block:: cmake
    :caption: src/AnalysisG/models/MyNewModel/CMakeLists.txt
    :linenos:

    # Define header and source files using your model's name
    set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/MyNewModel.h)
    set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/MyNewModel.cxx)

    # Add the C++ library target
    add_library(cMyNewModel STATIC ${SOURCE_FILES}) # Use c<model-name> convention

    # Set include directories for the C++ library
    target_include_directories(cMyNewModel PRIVATE include/models)
    target_include_directories(cMyNewModel PUBLIC include)

    # Link dependencies (core model template, add others if needed)
    target_link_libraries(cMyNewModel PUBLIC cmodel <dependencies>) # Add e.g. PyTorch::torch

    # Set compile options (typically needed for shared libraries/Python extensions)
    target_compile_options(cMyNewModel PRIVATE -fPIC)

    # Call the Cython build helper function
    # Arguments: <cython_module_path> <output_directory> <cpp_library_name> <python_class_name>
    cmake_language(CALL cybuild "models/MyNewModel" "models" cMyNewModel "PyMyNewModel")

**5. Rebuild the Project**

After making these changes, rebuild the `AnalysisG` project using CMake and your build system (e.g., ``make`` or ``ninja``) for the changes to take effect. Your new model (``PyMyNewModel``) should then be available for import and use in Python.

.. code-block:: bash

    cd <path_to_AnalysisG>/build
    cmake ..
    make # or ninja

You can then import your model in Python:

.. code-block:: python

    from AnalysisG.models import PyMyNewModel

    model = PyMyNewModel()
    print(model)
