.. _model-template:

The ModelTemplate
-----------------

This part of the documentation highlights some useful features that are part of the template class.
The model template class is useful for training and inference studies of a give machine learning model.
A main focus of this class is the application of GraphNeuralNetworks (GNNs).

The C++ Interface
^^^^^^^^^^^^^^^^^

.. cpp:class:: model_template: public notification, public tools

    .. cpp:function:: model_template()

       The default constructor of the model.

    .. cpp:function:: virtual ~model_template()

       The structor of the model.

    .. cpp:function:: virtual model_template* clone()

       Is a required function to clone the model multiple times.

    .. cpp:function:: virtual void forward(graph_t* data)

       The forward method used to implement the model architecture.

    .. cpp:function:: virtual void train_sequence(bool mode)

       A method used to define how the training sequence should be performed.
       By default, all the losses are aggregated into a single value, and then the optimizer minimizes this loss.

    .. cpp:function:: void check_features(graph_t*)

       Verifies that all the features are available.

    .. cpp:function:: void set_optimizer(std::string name)

       Sets the optimizer to be used.
       To define the optimizer, use the Python interfacing class, `from AnalysisG.core.lossfx import OptimizerConfig`.
       The optimizer configuration class is discussed further under the Python Interface class.

    .. cpp:function:: void initialize(optimizer_params_t*)

       An internal function used to initialize the model with given optimizer params.

    .. cpp:function:: void clone_settings(model_settings_t* setd)

       A fuction used to clone basic model configurations, such as the input/output parameters, device, model name, optimizer etc.

    .. cpp:function:: void import_settings(model_settings_t* setd)

       A function to import the model configurations.

    .. cpp:function:: void forward(graph_t* data, bool train)

       Similar to the other forward function, however it is used to indicate whether the model should be trained.
       Generally, this function does not require any further consideration.

    .. cpp:function:: void register_module(torch::nn::Sequential* data)

       A function which registers the torch::nn::Sequential architecture internally without any initial weight initialization.

    .. cpp:function:: void register_module(torch::nn::Sequential* data, mlp_init weight_init)

       A function which registers the torch::nn::Sequential architecture internally **with** initial weight initialization.

    .. cpp:function:: void prediction_graph_feature(std::string, torch::Tensor)

       A function registering the output prediction for a graph feature.
       It is important that the name of the output feature matches the truth feature during training.
       For instance, if the prediction is called `signal`, then the graph requires a truth name called `signal`.

    .. cpp:function:: void prediction_node_feature(std::string, torch::Tensor)

       A function registering the output prediction for a node feature.
       It is important that the name of the output feature matches the truth feature during training.
       For instance, if the prediction is called `signal`, then the graph requires a truth node name called `signal`.

    .. cpp:function:: void prediction_edge_feature(std::string, torch::Tensor)

       A function registering the output prediction for an edge feature.
       It is important that the name of the output feature matches the truth feature during training.
       For instance, if the prediction is called `some_edge`, then the graph requires a truth edge name called `some_edge`.

    .. cpp:function:: void prediction_extra(std::string, torch::Tensor)

       A function registering any additional output variables. 
       During training, this has no impact, but during inference, the variable is written as a leaf to ROOT files.

    .. cpp:function:: torch::Tensor compute_loss(std::string, graph_enum)

       Computes the loss associated with a particular feature, given either it is an edge, node, or graph feature (controlled by the `graph_enum`).

    .. cpp:function:: void evaluation_mode(bool mode = true)

    .. cpp:function:: void save_state()

    .. cpp:function:: bool restore_state()

    .. cpp:var:: cproperty<std::string, model_template> name

    .. cpp:var:: cproperty<std::string, model_template> device

    .. cpp:var:: int kfold

    .. cpp:var:: int epoch

    .. cpp:var:: bool use_pkl

    .. cpp:var:: bool inference_mode

    .. cpp:var:: std::string model_checkpoint_path

    .. cpp:var:: cproperty<std::map<std::string, std::string>, std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>> o_graph

    .. cpp:var:: cproperty<std::map<std::string, std::string>, std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>> o_node

    .. cpp:var:: cproperty<std::map<std::string, std::string>, std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>> o_edge

    .. cpp:var:: cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_graph

    .. cpp:var:: cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_node

    .. cpp:var:: cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_edge


The Python Interface
^^^^^^^^^^^^^^^^^^^^

.. py:class:: ModelTemplate

   .. py:attribute:: o_graph
      :type: dict

      Sets the output feature of the model and pairs the output with the associated loss function.

   .. py:attribute:: o_node
      :type: dict

      Sets the output feature of the model and pairs the output with the associated loss function.

   .. py:attribute:: o_edge
      :type: dict

      Sets the output feature of the model and pairs the output with the associated loss function.

   .. py:attribute:: i_graph
      :type: list

      Sets the input features.

   .. py:attribute:: i_node
      :type: list

      Sets the input features.

   .. py:attribute:: i_edge
      :type: list

      Sets the input features.

   .. py:attribute:: device
      :type: str

      Sets the device that the model should be using.
      Follows the standard syntax used by `PyTorch`, e.g. "cuda:0", "cuda:1"

   .. py:attribute:: checkpoint_path
      :type: str

      Path of the training checkpoint to use for model inference.


