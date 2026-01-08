.. default-domain:: cpp

.. file:: model_template.h

        Header file for the model_template class.

Includes
========

.. code-block:: cpp

        #include <notification/notification.h>
        #include <templates/graph_template.h>
        #include <templates/lossfx.h>
        #include <structs/settings.h>
        #include <structs/model.h>

        #ifdef PYC_CUDA
        #include <c10/cuda/CUDAStream.h>
        #include <ATen/cuda/CUDAGraph.h>
        #endif

Forward Declarations
====================

.. code-block:: cpp

        class metrics;
        class analysis;
        class model_template;
        class metric_template;
        class optimizer;
        class dataloader;

        struct graph_t;
        struct variable_t;
        struct optimizer_params_t;
        struct model_report;

Class Definition
================

.. cpp:class:: model_template : public notification, public tools

        Public Members
        --------------

        .. cpp:function:: model_template()

                model_template Funktion

                Detaillierte Beschreibung der model_template Funktion

        .. cpp:function:: virtual ~model_template()

                ~model_template Funktion

                Detaillierte Beschreibung der ~model_template Funktion

        .. cpp:member:: cproperty<int, model_template> device_index

                Set device of model.

        .. cpp:member:: cproperty<std::string, model_template> name

        .. cpp:member:: cproperty<std::string, model_template> device

        .. cpp:member:: int kfold

                Model state.

        .. cpp:member:: int epoch

        .. cpp:member:: bool is_mc = false

        .. cpp:member:: bool use_pkl = false

        .. cpp:member:: bool inference_mode = false

        .. cpp:member:: std::string model_checkpoint_path = ""

        .. cpp:member:: std::string weight_name = "event_weight"

        .. cpp:member:: std::string tree_name = "nominal"

        .. cpp:member:: cproperty<std::map<std::string, std::string>, std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>> o_graph

                Target properties for each graph object: name - loss.

        .. cpp:member:: cproperty<std::map<std::string, std::string>, std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>> o_node

        .. cpp:member:: cproperty<std::map<std::string, std::string>, std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>> o_edge

        .. cpp:member:: cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_graph

                Requested input features.

        .. cpp:member:: cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_node

        .. cpp:member:: cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_edge

        .. cpp:function:: virtual model_template* clone()

        .. cpp:function:: virtual void forward(graph_t* data)

        .. cpp:function:: virtual void train_sequence(bool mode)

        .. cpp:function:: void check_features(graph_t*)

                check_features Funktion

                Detaillierte Beschreibung der check_features Funktion

        .. cpp:function:: void set_optimizer(std::string name)

                set_optimizer Funktion

                Detaillierte Beschreibung der set_optimizer Funktion

        .. cpp:function:: void initialize(optimizer_params_t*)

                initialize Funktion

                Detaillierte Beschreibung der initialize Funktion

        .. cpp:function:: void clone_settings(model_settings_t* setd)

                clone_settings Funktion

                Detaillierte Beschreibung der clone_settings Funktion

        .. cpp:function:: void import_settings(model_settings_t* setd)

                import_settings Funktion

                Detaillierte Beschreibung der import_settings Funktion

        .. cpp:function:: void forward(graph_t* data, bool train)

                forward Funktion

                Detaillierte Beschreibung der forward Funktion

        .. cpp:function:: void forward(std::vector<graph_t*> data, bool train)

                forward Funktion

                Detaillierte Beschreibung der forward Funktion

        .. cpp:function:: void register_module(torch::nn::Sequential* data)

                register_module Funktion

                Detaillierte Beschreibung der register_module Funktion

        .. cpp:function:: void register_module(torch::nn::Sequential* data, mlp_init weight_init)

                register_module Funktion

                Detaillierte Beschreibung der register_module Funktion

        .. cpp:function:: void prediction_graph_feature(std::string, torch::Tensor)

                prediction_graph_feature Funktion

                Detaillierte Beschreibung der prediction_graph_feature Funktion

        .. cpp:function:: void prediction_node_feature(std::string, torch::Tensor)

                prediction_node_feature Funktion

                Detaillierte Beschreibung der prediction_node_feature Funktion

        .. cpp:function:: void prediction_edge_feature(std::string, torch::Tensor)

                prediction_edge_feature Funktion

                Detaillierte Beschreibung der prediction_edge_feature Funktion

        .. cpp:function:: void prediction_extra(std::string, torch::Tensor)

                prediction_extra Funktion

                Detaillierte Beschreibung der prediction_extra Funktion

        .. cpp:function:: torch::Tensor* compute_loss(std::string, graph_enum)

                compute_loss Funktion

                Detaillierte Beschreibung der compute_loss Funktion

        .. cpp:function:: void evaluation_mode(bool mode = true)

                evaluation_mode Funktion

                Detaillierte Beschreibung der evaluation_mode Funktion

        .. cpp:function:: void save_state()

                save_state Funktion

                Detaillierte Beschreibung der save_state Funktion

        .. cpp:function:: bool restore_state()

                restore_state Funktion

                Detaillierte Beschreibung der restore_state Funktion

        Friends
        -------

        .. cpp:friend:: struct graph_t
        .. cpp:friend:: class metrics
        .. cpp:friend:: class analysis
        .. cpp:friend:: class optimizer
        .. cpp:friend:: class dataloader
        .. cpp:friend:: class metric_template

        Private Members
        ---------------

        .. cpp:function:: model_template* clone(int)

                clone Funktion

                Detaillierte Beschreibung der clone Funktion

        .. cpp:staticfunction:: void set_input_features(std::vector<std::string>*, std::map<std::string, torch::Tensor*>*)

        .. cpp:staticfunction:: void set_output_features(std::map<std::string, std::string>*, std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>*)

        .. cpp:staticfunction:: void get_name(std::string*, model_template*)

        .. cpp:staticfunction:: void set_name(std::string*, model_template*)

        .. cpp:staticfunction:: void get_dev_index(int*, model_template*)

        .. cpp:staticfunction:: void set_dev_index(int*, model_template*)

        .. cpp:function:: template <typename G, typename F> void assign(std::map<std::string, G>* inpt, graph_enum mode, F* data)

                assign Funktion

                Detaillierte Beschreibung der assign Funktion

        .. cpp:staticfunction:: void set_device(std::string*, model_template*)

        .. cpp:function:: torch::Tensor* assign_features(std::string inpt, graph_enum type, graph_t* data)

                assign_features Funktion

                Detaillierte Beschreibung der assign_features Funktion

        .. cpp:function:: torch::Tensor* assign_features(std::string inpt, graph_enum type, std::vector<graph_t*>* data)

                assign_features Funktion

                Detaillierte Beschreibung der assign_features Funktion

        .. cpp:function:: void flush_outputs()

                flush_outputs Funktion

                Detaillierte Beschreibung der flush_outputs Funktion

        .. cpp:member:: lossfx* m_loss = nullptr
        .. cpp:member:: torch::TensorOptions* m_option = nullptr
        .. cpp:member:: torch::optim::Optimizer* m_optim = nullptr
        .. cpp:member:: torch::Tensor* edge_index = nullptr
        .. cpp:member:: bool m_batched = false
        .. cpp:member:: int m_device_idx = -2
        .. cpp:member:: opt_enum e_optim = opt_enum::invalid_optimizer
        .. cpp:member:: std::string s_optim = ""
        .. cpp:member:: std::string m_name = "model-template"
        .. cpp:member:: std::vector<torch::nn::Sequential*> m_data = {}
        .. cpp:member:: std::vector<torch::Tensor*> _losses = {}
        .. cpp:member:: std::map<std::string, torch::Tensor*> m_i_graph = {}
        .. cpp:member:: std::map<std::string, torch::Tensor*> m_i_node = {}
        .. cpp:member:: std::map<std::string, torch::Tensor*> m_i_edge = {}
        .. cpp:member:: std::map<std::string, torch::Tensor*> m_p_graph = {}
        .. cpp:member:: std::map<std::string, torch::Tensor*> m_p_node = {}
        .. cpp:member:: std::map<std::string, torch::Tensor*> m_p_edge = {}
        .. cpp:member:: std::map<std::string, torch::Tensor*> m_p_undef = {}
        .. cpp:member:: std::map<std::string, std::tuple<torch::Tensor*, loss_enum>> m_o_graph = {}
        .. cpp:member:: std::map<std::string, std::tuple<torch::Tensor*, loss_enum>> m_o_node = {}
        .. cpp:member:: std::map<std::string, std::tuple<torch::Tensor*, loss_enum>> m_o_edge = {}
        .. cpp:member:: std::map<graph_enum, std::map<std::string, torch::Tensor>> m_p_loss = {}
