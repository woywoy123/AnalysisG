.. _model-template:

ModelTemplate Methods
---------------------

This part of the documentation highlights some useful features that are part of the template class.
The model template class is useful for training and inference studies of a give machine learning model.
A main focus of this class is the application of GraphNeuralNetworks (GNNs).


.. cpp:class:: model_template: public notification, public tools

    .. cpp:function:: model_template()

    .. cpp:function:: virtual ~model_template()

    .. cpp:function:: virtual model_template* clone()

    .. cpp:function:: virtual void forward(graph_t* data)

    .. cpp:function:: virtual void train_sequence(bool mode)

    .. cpp:function:: void check_features(graph_t*)

    .. cpp:function:: void set_optimizer(std::string name)

    .. cpp:function:: void initialize(optimizer_params_t*)

    .. cpp:function:: void clone_settings(model_settings_t* setd)

    .. cpp:function:: void import_settings(model_settings_t* setd)

    .. cpp:function:: void forward(graph_t* data, bool train)

    .. cpp:function:: void register_module(torch::nn::Sequential* data)

    .. cpp:function:: void prediction_graph_feature(std::string, torch::Tensor)

    .. cpp:function:: void prediction_node_feature(std::string, torch::Tensor)

    .. cpp:function:: void prediction_edge_feature(std::string, torch::Tensor)

    .. cpp:function:: void prediction_extra(std::string, torch::Tensor)

    .. cpp:function:: torch::Tensor compute_loss(std::string, graph_enum)

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

