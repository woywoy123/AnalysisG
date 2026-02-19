Model Template Module (C++)
============================

The Model Template module provides the C++ implementation for ML models.

Overview
--------

Located in ``src/AnalysisG/modules/model/``, this module implements model template 
functionality in C++:

- Neural network model interface
- Integration with LibTorch
- Model training and inference
- State serialization

Purpose
-------

The model module enables:

- Defining custom neural network architectures
- Training on graph/tensor data
- Inference on new data
- Model checkpointing

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/model/cxx/*.cxx`` - Model implementations
- ``src/AnalysisG/modules/model/include/templates/*.h`` - Model template headers

**Python Binding**

- ``src/AnalysisG/core/model_template.pyx`` - Cython wrapper
- ``src/AnalysisG/core/model_template.pxd`` - Cython declarations

Key Classes
-----------

**model_template**

Base template for ML models:

.. code-block:: cpp

   class model_template {
   public:
       // Model configuration
       cproperty<std::vector<std::string>, model_template> i_graph;  // Input graph features
       cproperty<std::vector<std::string>, model_template> i_node;   // Input node features
       cproperty<std::vector<std::string>, model_template> i_edge;   // Input edge features
       
       cproperty<std::vector<std::string>, model_template> o_graph;  // Output graph features
       cproperty<std::vector<std::string>, model_template> o_node;   // Output node features
       cproperty<std::vector<std::string>, model_template> o_edge;   // Output edge features
       
       cproperty<std::string, model_template> device;                // "cpu" or "cuda:0"
       cproperty<std::string, model_template> name;                  // Model name
       
       // Training state
       cproperty<bool, model_template> training;                     // Training mode
       cproperty<int, model_template> epoch;                         // Current epoch
       
       // Virtual methods
       virtual torch::Tensor forward(torch::Tensor x);               // Forward pass
       virtual void train();                                         // Set training mode
       virtual void eval();                                          // Set evaluation mode
       
       // Model management
       virtual void save(std::string path);                          // Save model
       virtual void load(std::string path);                          // Load model
       virtual model_template* clone();                              // Clone model
   };

LibTorch Integration
--------------------

Models are implemented using LibTorch (PyTorch C++ API):

**Forward Pass**

.. code-block:: cpp

   torch::Tensor forward(torch::Tensor x) override {
       // Define forward computation
       x = linear1->forward(x);
       x = torch::relu(x);
       x = linear2->forward(x);
       return x;
   }

**Layers**

Common layer types:

- Linear: ``torch::nn::Linear``
- Convolutional: ``torch::nn::Conv2d``
- Graph convolution: Custom GNN layers
- Batch normalization: ``torch::nn::BatchNorm1d``
- Dropout: ``torch::nn::Dropout``

Usage Example
-------------

**Defining a Model**

.. code-block:: cpp

   #include <templates/model_template.h>
   
   class GNNModel : public model_template {
   public:
       // Network layers
       torch::nn::Linear conv1{nullptr};
       torch::nn::Linear conv2{nullptr};
       torch::nn::Linear fc{nullptr};
       
       GNNModel() {
           // Initialize layers
           conv1 = register_module("conv1", torch::nn::Linear(64, 128));
           conv2 = register_module("conv2", torch::nn::Linear(128, 128));
           fc = register_module("fc", torch::nn::Linear(128, 2));
           
           // Set device
           device = "cuda:0";
           name = "GNN_Classifier";
       }
       
       torch::Tensor forward(torch::Tensor x) override {
           x = torch::relu(conv1->forward(x));
           x = torch::relu(conv2->forward(x));
           x = fc->forward(x);
           return x;
       }
   };

**Using the Model**

.. code-block:: cpp

   // Create model
   GNNModel model;
   model.to(torch::kCUDA);
   
   // Training mode
   model.train();
   
   // Forward pass
   auto output = model.forward(input_tensor);
   
   // Compute loss
   auto loss = torch::nn::functional::cross_entropy(output, targets);
   
   // Backward pass
   loss.backward();
   
   // Optimizer step
   optimizer.step();
   
   // Save model
   model.save("checkpoint.pt");

Model Architecture Patterns
----------------------------

**Graph Neural Networks**

GNN for particle physics:

.. code-block:: cpp

   class ParticleGNN : public model_template {
       // Node encoder
       torch::nn::Linear node_encoder;
       
       // Graph convolution layers
       std::vector<GraphConv> graph_convs;
       
       // Global pooling
       GlobalAttentionPooling global_pool;
       
       // Classifier
       torch::nn::Linear classifier;
       
       torch::Tensor forward(torch::Tensor node_features, 
                            torch::Tensor edge_index) override {
           // Encode nodes
           auto h = node_encoder->forward(node_features);
           
           // Graph convolutions
           for (auto& conv : graph_convs) {
               h = conv.forward(h, edge_index);
               h = torch::relu(h);
           }
           
           // Global pooling
           auto g = global_pool.forward(h, batch_index);
           
           // Classify
           return classifier->forward(g);
       }
   };

**Recurrent Neural Networks**

RNN for sequential data:

.. code-block:: cpp

   class SequenceModel : public model_template {
       torch::nn::LSTM lstm;
       torch::nn::Linear fc;
       
       torch::Tensor forward(torch::Tensor sequence) override {
           auto lstm_out = lstm->forward(sequence);
           auto output = fc->forward(lstm_out);
           return output;
       }
   };

Training Integration
--------------------

**Training Loop**

.. code-block:: cpp

   // Setup
   model.train();
   auto optimizer = torch::optim::Adam(model.parameters(), 0.001);
   
   // Training loop
   for (int epoch = 0; epoch < num_epochs; ++epoch) {
       for (auto& batch : dataloader) {
           // Zero gradients
           optimizer.zero_grad();
           
           // Forward pass
           auto output = model.forward(batch.x);
           
           // Compute loss
           auto loss = criterion(output, batch.y);
           
           // Backward pass
           loss.backward();
           
           // Update weights
           optimizer.step();
       }
   }

**Inference**

.. code-block:: cpp

   // Set evaluation mode
   model.eval();
   
   // Disable gradient computation
   torch::NoGradGuard no_grad;
   
   // Inference
   for (auto& batch : test_loader) {
       auto output = model.forward(batch.x);
       auto predictions = torch::argmax(output, 1);
   }

Model Persistence
-----------------

**Saving Models**

.. code-block:: cpp

   // Save full model
   model.save("model_checkpoint.pt");
   
   // Save state dict only
   torch::save(model.parameters(), "model_state.pt");

**Loading Models**

.. code-block:: cpp

   // Load full model
   model.load("model_checkpoint.pt");
   
   // Load state dict
   torch::load(model.parameters(), "model_state.pt");

Integration with Python
-----------------------

The C++ model_template is wrapped in Python:

.. code-block:: python

   from AnalysisG.core.model_template import ModelTemplate
   
   class MyModel(ModelTemplate):
       def __init__(self):
           super().__init__()
           self.device = "cuda:0"
           self.i_node = ["pt", "eta", "phi", "energy"]
           self.o_graph = ["signal_prob"]

Device Management
-----------------

**GPU Support**

.. code-block:: cpp

   // Check CUDA availability
   if (torch::cuda::is_available()) {
       model.device = "cuda:0";
       model.to(torch::kCUDA);
   }

**Multi-GPU**

.. code-block:: cpp

   // Data parallel
   model.device = "cuda";
   auto parallel_model = torch::nn::DataParallel(model);

See Also
--------

* :doc:`../core/model_template` - Python ModelTemplate wrapper
* :doc:`metric` - Metric template for evaluation
* :doc:`lossfx` - Loss functions and optimizers
* :doc:`dataloader` - Data loading for training
