The ModelTemplate Source Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a new model to the framework, navigate to the template files (src/AnalysisG/templates/model) and copy the source files into the existing model directory.
Make sure to rename the folder appropriately, and rename files with the `<model-name>` prefix as needed.
The given template files have all the needed structures in place, and essentially just require search and replace modifications.

The C++ Source Code
^^^^^^^^^^^^^^^^^^^

.. code:: C++
    
   #ifndef <model-name>_H
   #define <model-name>_H
   #include <templates/model_template.h>
   
   class <model-name>: public model_template
   {
       public:
           <model-name>();
           ~<model-name>();
           model_template* clone() override;
           void forward(graph_t*) override; 
   
           torch::nn::Sequential* example = nullptr; 
   }; 
   
   #endif


.. code:: C++
  
   #include <model.h>
   
   model::model(){
   
       this -> example = new torch::nn::Sequential({
               {"L1", torch::nn::Linear(2, 2)},
               {"RELU", torch::nn::ReLU()},
               {"L2", torch::nn::Linear(2, 2)}
       }); 
   
       this -> register_module(this -> example); 
   }
   
   void model::forward(graph_t* data){
   
       // fetch the input data of the model.
       // If the variable is not available, this will return a nullptr.
       torch::Tensor graph = data -> get_data_graph("graph") -> clone(); 
       torch::Tensor node  = data -> get_data_node("node") -> clone();
       torch::Tensor edge  = data -> get_data_edge("edge") -> clone(); 
       torch::Tensor edge_index = data -> edge_index -> clone(); 
   
       // output the prediction weights for edges, nodes, graphs.
       this -> prediction_graph_feature("..."; <some-tensor>); 
       this -> prediction_node_feature("...", <some-tensor>);
       this -> prediction_edge_feature("...", <some-tensor>); 
       if (!this -> inference_mode){return;} // skips any variables not avaliable during inference time.
       this -> prediction_extra("...", <some-tensor>);  // Any variables that should be dumped during the inference.
   }
   
   model::~model(){}
   model_template* model::clone(){
       return new model(); 
   }

Cython Interface Files
^^^^^^^^^^^^^^^^^^^^^^

The code below would be the interface of the model via Cython which can be initialized from the Python interpreter
Similar to C++, they require a header (.pxd) and source file (.pyx).

.. py:class:: model.pxd

   # distutils: language=c++
   # cython: language_level=3
   
   from libcpp cimport int, bool
   from AnalysisG.core.model_template cimport model_template, ModelTemplate
   
   cdef extern from "<models/model.h>":
       cdef cppclass model(model_template):
           model() except+
   
   cdef class ExampleModel(ModelTemplate): pass
    
.. py:class:: model.pyx

   # distutils: language=c++
   # cython: language_level=3
   
   from AnalysisG.core.model_template cimport ModelTemplate
   from AnalysisG.models.<model-name> cimport ExampleModel
   
   cdef class ExampleModel(ModelTemplate):
       def __cinit__(self): self.nn_ptr = new model()
       def __init__(self): pass
       def __dealloc__(self): del self.nn_ptr
    
