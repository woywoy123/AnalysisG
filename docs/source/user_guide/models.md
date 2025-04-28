# Machine Learning Models

This section describes how to define, configure, and use machine learning models within the AnalysisG framework.

## Model Architecture

AnalysisG provides a flexible system for defining machine learning models, with special emphasis on Graph Neural Networks (GNNs). The framework integrates with PyTorch for model definition and training.

### model_template Class

The `model_template` class serves as the base class for all models in AnalysisG:

```cpp
model_template* model = new model_template();
```

This class provides:
- Integration with PyTorch's module system
- Methods for forward and backward passes
- Optimization configuration
- Model persistence (saving/loading)

## Predefined Models

AnalysisG comes with several predefined model architectures:

### Graph Neural Network (GNN)

```cpp
model_template* create_gnn_model() {
    model_template* model = new model_template();
    
    // Define a Graph Neural Network
    model->add_module("gnn", torch::nn::ModuleDict({
        {"node_encoder", torch::nn::Linear(node_dim, hidden_dim)},
        {"edge_encoder", torch::nn::Linear(edge_dim, hidden_dim)},
        
        {"graph_conv1", GraphConv(hidden_dim, hidden_dim)},
        {"graph_conv2", GraphConv(hidden_dim, hidden_dim)},
        {"graph_conv3", GraphConv(hidden_dim, hidden_dim)},
        
        {"graph_pooling", GlobalMeanPool()},
        {"classifier", torch::nn::Linear(hidden_dim, output_dim)}
    }));
    
    return model;
}
```

### Multi-Layer Perceptron (MLP)

```cpp
model_template* create_mlp_model() {
    model_template* model = new model_template();
    
    // Define a Multi-Layer Perceptron
    model->add_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(input_dim, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Dropout(dropout_prob),
        torch::nn::Linear(hidden_dim, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Dropout(dropout_prob),
        torch::nn::Linear(hidden_dim, output_dim)
    ));
    
    return model;
}
```

## Custom Models

You can define custom models by implementing your own model class:

```cpp
class CustomGNN : public model_template {
public:
    CustomGNN(int node_dim, int edge_dim, int hidden_dim, int output_dim) {
        // Define custom layers
        node_encoder = register_module("node_encoder", 
            torch::nn::Linear(node_dim, hidden_dim));
        
        edge_encoder = register_module("edge_encoder", 
            torch::nn::Linear(edge_dim, hidden_dim));
            
        conv1 = register_module("conv1", 
            GraphConv(hidden_dim, hidden_dim));
        
        conv2 = register_module("conv2", 
            GraphConv(hidden_dim, hidden_dim));
            
        pooling = register_module("pooling", 
            GlobalAttentionPool(hidden_dim));
            
        classifier = register_module("classifier", 
            torch::nn::Linear(hidden_dim, output_dim));
    }
    
    torch::Tensor forward(graph_data_t& data) {
        // Implement forward pass
        torch::Tensor x = node_encoder->forward(data.node_features);
        torch::Tensor edge_attr = edge_encoder->forward(data.edge_features);
        
        x = conv1->forward(x, data.edge_index, edge_attr);
        x = torch::relu(x);
        
        x = conv2->forward(x, data.edge_index, edge_attr);
        x = torch::relu(x);
        
        x = pooling->forward(x, data.batch);
        x = classifier->forward(x);
        
        return x;
    }
    
private:
    torch::nn::Linear node_encoder{nullptr}, edge_encoder{nullptr}, classifier{nullptr};
    GraphConv conv1{nullptr}, conv2{nullptr};
    GlobalAttentionPool pooling{nullptr};
};
```

## Model Configuration

Models can be configured using settings:

```cpp
void configure_model(model_template* model, settings_t* settings) {
    // Set the optimizer
    model->set_optimizer(settings->optimizer);
    
    // Configure optimizer parameters
    optimizer_params_t opt_params;
    opt_params.learning_rate = settings->learning_rate;
    opt_params.weight_decay = settings->weight_decay;
    opt_params.momentum = settings->momentum;
    
    // Initialize the model
    model->initialize(&opt_params);
    
    // Set loss function
    if (settings->loss_function == "cross_entropy") {
        model->set_loss_function(torch::nn::CrossEntropyLoss());
    } else if (settings->loss_function == "bce") {
        model->set_loss_function(torch::nn::BCEWithLogitsLoss());
    }
}
```

## Training a Model

You can train a model using the AnalysisG framework:

```cpp
void train_model(model_template* model, std::vector<graph_t*>* training_data) {
    // Configure training settings
    train_params_t params;
    params.batch_size = 64;
    params.epochs = 100;
    params.shuffle = true;
    params.validation_fraction = 0.2;
    
    // Start training
    model->train(&params, training_data);
    
    // Training callbacks are also supported
    model->set_epoch_callback([](model_template* m, int epoch, float loss) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    });
}
```

## Model Evaluation

After training, you can evaluate the model's performance:

```cpp
void evaluate_model(model_template* model, std::vector<graph_t*>* test_data) {
    // Perform evaluation
    eval_result_t results = model->evaluate(test_data);
    
    // Print evaluation metrics
    std::cout << "Accuracy: " << results.accuracy << std::endl;
    std::cout << "ROC AUC: " << results.roc_auc << std::endl;
    
    // Generate evaluation plots
    plotting* plot = new plotting();
    plot->set_data(results.predictions, results.truth_labels);
    plot->roc_curve("model_roc.pdf");
    plot->confusion_matrix("confusion_matrix.pdf");
    delete plot;
}
```

## Model Persistence

Models can be saved and loaded:

```cpp
// Save model
model->save_state("model_checkpoint.pt");

// Load model
model_template* loaded_model = new model_template();
loaded_model->restore_state("model_checkpoint.pt");
```

## Integration with Analysis Module

Models are typically used as part of a complete analysis:

```cpp
analysis* an = new analysis();

// Configure the analysis
settings_t settings;
settings.model_name = "GNN";
settings.learning_rate = 0.001;
settings.epochs = 50;
an->import_settings(&settings);

// Set the model template
an->set_model_template([](const settings_t* settings) -> model_template* {
    return create_gnn_model();
});

// Run the analysis
an->run();
```

## Best Practices

1. **Appropriate Architecture**: Choose a model architecture suitable for your physics problem
2. **Hyperparameter Tuning**: Use cross-validation to find optimal hyperparameters
3. **Regularization**: Apply dropout and weight decay to prevent overfitting
4. **Batch Size**: Adjust batch size based on available GPU memory
5. **Learning Rate**: Use an appropriate learning rate schedule for convergence