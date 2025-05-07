/**
 * @file analysis.h
 * @brief Defines the `analysis` class, the central orchestrator for physics analysis tasks.
 *
 * This file declares the `analysis` class, which manages the overall workflow of a physics analysis.
 * It handles the setup of event data, selections, graph representations, machine learning models,
 * and metrics. It also orchestrates the execution of the analysis, including event processing,
 * model training/inference, and result aggregation. The class inherits from `notification` and `tools`
 * for logging and utility functions, respectively.
 */

#ifndef ANALYSIS_H ///< Start of include guard for ANALYSIS_H to prevent multiple inclusions.
#define ANALYSIS_H ///< Definition of ANALYSIS_H to signify the header has been included.

#include <string> ///< Includes the standard string library for string manipulation.
#include <generators/sampletracer.h> ///< Includes the `sampletracer` class for tracking sample processing.
#include <generators/dataloader.h> ///< Includes the `dataloader` class for managing data loading.
#include <generators/optimizer.h> ///< Includes the `optimizer` class for model training.

#include <templates/graph_template.h> ///< Includes the base template for graph structures.
#include <templates/event_template.h> ///< Includes the base template for event data.
#include <templates/metric_template.h> ///< Includes the base template for performance metrics.
#include <templates/selection_template.h> ///< Includes the base template for event selections.
#include <templates/model_template.h> ///< Includes the base template for machine learning models.
#include <structs/settings.h> ///< Includes the structure for analysis-wide settings.
#include <io/io.h> ///< Includes the `io` class for input/output operations.

/**
 * @class analysis
 * @brief Orchestrates the entire physics analysis workflow.
 *
 * The `analysis` class is the main driver for performing a physics analysis. It manages
 * the configuration, data processing, model execution, and result collection. Users interact
 * with this class to define the analysis steps, add samples, specify models, and run the analysis.
 */
class analysis: // Defines the class 'analysis'.
    public notification, ///< Inherits from the `notification` class for logging capabilities.
    public tools ///< Inherits from the `tools` class for utility functions.
{
public: ///< Public access specifier for the following members.
    /**
     * @brief Constructor for the `analysis` class.
     * Initializes a new analysis instance.
     */
    analysis();
    /**
     * @brief Destructor for the `analysis` class.
     * Cleans up resources, such as deleting dynamically allocated objects.
     */
    ~analysis(); 

    /**
     * @brief Adds data samples to the analysis.
     * @param path Path to the data sample (file or directory).
     * @param label A label to identify this sample.
     */
    void add_samples(std::string path, std::string label);
    /**
     * @brief Adds a selection template to the analysis.
     * @param sel Pointer to a `selection_template` object.
     */
    void add_selection_template(selection_template* sel); 
    /**
     * @brief Adds an event template to the analysis, associating it with a label.
     * @param ev Pointer to an `event_template` object.
     * @param label A label to identify this event template.
     */
    void add_event_template(event_template* ev, std::string label); 
    /**
     * @brief Adds a graph template to the analysis, associating it with a label.
     * @param gr Pointer to a `graph_template` object.
     * @param label A label to identify this graph template.
     */
    void add_graph_template(graph_template* gr, std::string label); 
    /**
     * @brief Adds a metric template to the analysis, associating it with a model.
     * @param mx Pointer to a `metric_template` object.
     * @param mdl Pointer to a `model_template` object that this metric applies to.
     */
    void add_metric_template(metric_template* mx, model_template* mdl);
    /**
     * @brief Adds a machine learning model and its optimizer parameters to the analysis for training.
     * @param model Pointer to a `model_template` object.
     * @param op Pointer to an `optimizer_params_t` object containing optimizer settings.
     * @param run_name A name for this training run or model session.
     */
    void add_model(model_template* model, optimizer_params_t* op, std::string run_name); 
    /**
     * @brief Adds a machine learning model to the analysis, typically for inference or metrics without explicit optimizer params here.
     * @param model Pointer to a `model_template` object.
     * @param run_name A name for this model session (e.g., for inference or pre-trained model metrics).
     */
    void add_model(model_template* model, std::string run_name); 
    /**
     * @brief Attaches or manages threads for parallel processing, if implemented.
     */
    void attach_threads(); 
    /**
     * @brief Starts the analysis execution.
     * This method triggers the main analysis loop, including data processing, model training/inference, etc.
     */
    void start(); 

    /**
     * @brief Retrieves the progress of different analysis tasks or model trainings.
     * @return A map where keys are task/model names and values are vectors of floats representing progress (e.g., percentage or epoch number).
     */
    std::map<std::string, std::vector<float>> progress(); 
    /**
     * @brief Retrieves the current mode (e.g., training, evaluating) of different analysis tasks.
     * @return A map where keys are task/model names and values are strings describing the mode.
     */
    std::map<std::string, std::string> progress_mode(); 
    /**
     * @brief Retrieves a textual report of the progress or status of different analysis tasks.
     * @return A map where keys are task/model names and values are strings containing progress reports.
     */
    std::map<std::string, std::string> progress_report(); 
    /**
     * @brief Checks if specific analysis tasks or model trainings are complete.
     * @return A map where keys are task/model names and values are booleans (true if complete, false otherwise).
     */
    std::map<std::string, bool> is_complete();

    settings_t m_settings; ///< Object holding the analysis-wide settings and configurations.
    std::map<std::string, meta*> meta_data = {}; ///< Map storing metadata objects, keyed by a string identifier (e.g., sample label). Initialized empty.

private: ///< Private access specifier for the following members.
    /**
     * @brief Checks for cached data and potentially loads it to speed up processing.
     */
    void check_cache(); 
    /**
     * @brief Builds the project structure, possibly setting up directories or initial configurations.
     */
    void build_project(); 
    /**
     * @brief Builds or processes event data based on the added event templates and samples.
     */
    void build_events(); 
    /**
     * @brief Builds or applies event selections based on the added selection templates.
     */
    void build_selections(); 
    /**
     * @brief Builds graph representations of the event data based on the added graph templates.
     */
    void build_graphs(); 
    /**
     * @brief Sets up and configures machine learning model sessions for training or inference.
     */
    void build_model_session(); 
    /**
     * @brief Sets up the environment for model inference (prediction).
     */
    void build_inference();

    /**
     * @brief Builds or calculates metrics for evaluating model performance.
     * @return True if metric building was successful, false otherwise.
     */
    bool build_metric(); 
    /**
     * @brief Builds or calculates metrics across different data folds (e.g., for k-fold cross-validation).
     */
    void build_metric_folds();

    /**
     * @brief Builds or configures the data loader for training or evaluation.
     * @param training Boolean flag, true if for training, false if for evaluation/inference.
     */
    void build_dataloader(bool training); 
    /**
     * @brief Fetches tags, possibly related to data versions, processing stages, or k-folds.
     */
    void fetchtags(); 
    bool started = false;  ///< Flag indicating whether the analysis has been started. Initialized to false.

    /**
     * @brief Static helper method to add content (features) to a data map from various sources.
     * This version likely handles adding features from a TTree to torch::Tensor objects.
     * @param data Pointer to a map where keys are feature names and values are `torch::Tensor` pointers.
     * @param content Pointer to a vector of `variable_t` structs describing the features to extract.
     * @param index Current event index or entry number.
     * @param prefx A prefix string for feature names.
     * @param tt Pointer to the TTree to read from (optional, defaults to nullptr).
     * @return An integer status code (e.g., 0 for success).
     */
    static int add_content(
        std::map<std::string, torch::Tensor*>* data, 
        std::vector<variable_t>* content, int index, 
        std::string prefx, TTree* tt = nullptr
    ); 

    /**
     * @brief Static helper method to add content (features) to a data map from a buffer of tensors.
     * This version likely handles adding features from pre-loaded tensor buffers.
     * @param data Pointer to a map where keys are feature names and values are `torch::Tensor` pointers.
     * @param buff Pointer to a vector of vectors of `torch::Tensor` (the buffer).
     * @param edge Pointer to a `torch::Tensor` object for edge features.
     * @param node Pointer to a `torch::Tensor` object for node features.
     * @param batch Pointer to a `torch::Tensor` object for batch information.
     * @param mask Vector of long integers representing a mask.
     */
    static void add_content(
        std::map<std::string, torch::Tensor*>* data, std::vector<std::vector<torch::Tensor>>* buff, 
        torch::Tensor* edge, torch::Tensor* node, torch::Tensor* batch, std::vector<long> mask
    ); 

    /**
     * @brief Static method to execute a model (training or inference) on a set of data.
     * @param mdx Pointer to the `model_template` to be executed.
     * @param mds `model_settings_t` object containing settings for this model execution.
     * @param data Pointer to a vector of `graph_t` objects (input data).
     * @param prg Pointer to a size_t variable for progress tracking.
     * @param output Path or identifier for the output.
     * @param content Pointer to a vector of `variable_t` for output content description.
     * @param msg Pointer to a string for status messages.
     */
    static void execution(
        model_template* mdx, model_settings_t mds, std::vector<graph_t*>* data, size_t* prg,
        std::string output, std::vector<variable_t>* content, std::string* msg
    );

    /**
     * @brief Static method to execute metric calculation.
     * @param mt Pointer to a `metric_t` object (likely containing metric definitions and results).
     * @param prg Pointer to a size_t variable for progress tracking.
     * @param msg Pointer to a string for status messages.
     */
    static void execution_metric(metric_t* mt, size_t* prg, std::string* msg); 

    /**
     * @brief Static method to initialize a training loop for an optimizer.
     * @param op Pointer to the `optimizer` object.
     * @param k The k-fold index or a similar loop identifier.
     * @param model Pointer to the `model_template` being trained.
     * @param config Pointer to the `optimizer_params_t` for this training session.
     * @param rep Double pointer to a `model_report` object to store training reports.
     */
    static void initialize_loop(
        optimizer* op, int k, model_template* model, 
        optimizer_params_t* config, model_report** rep
    );

    /**
     * @brief Templated helper method to safely clone objects (e.g., event or graph templates).
     * @tparam g The type of the object to clone.
     * @param mp Pointer to a map where the cloned object might be stored or looked up.
     * @param in Pointer to the object to be cloned.
     */
    template <typename g>
    void safe_clone(std::map<std::string, g*>* mp, g* in){
        std::string name = in -> name; 
        if (mp -> count(name)){return;}
        (*mp)[name] = in -> clone(1); 
    }

    std::map<std::string, std::string> file_labels = {}; ///< Map to associate file paths or identifiers with user-defined labels. Initialized empty.
    std::map<std::string, event_template*> event_labels = {}; ///< Map storing event templates, keyed by a label. Initialized empty.
    std::map<std::string, metric_template*> metric_names = {}; ///< Map storing metric templates, keyed by a name or identifier. Initialized empty.
    std::map<std::string, selection_template*> selection_names = {}; ///< Map storing selection templates, keyed by a name or identifier. Initialized empty.
    std::map<std::string, std::map<std::string, graph_template*>> graph_labels = {}; ///< Map storing graph templates, nested by two string keys (e.g., category and label). Initialized empty.

    std::vector<std::string> model_session_names = {}; ///< Vector storing names of model sessions (e.g., for different training runs). Initialized empty.
    std::map<std::string, model_template*> model_inference = {}; ///< Map storing models configured for inference, keyed by a run name. Initialized empty.
    std::map<std::string, model_template*> model_metrics   = {}; ///< Map storing models for which metrics are to be calculated, keyed by a run name. Initialized empty.
    std::vector<std::tuple<model_template*, optimizer_params_t*>> model_sessions = {}; ///< Vector storing tuples of model templates and their optimizer parameters for training sessions. Initialized empty.

    std::map<std::string, optimizer*> trainer = {}; ///< Map storing optimizer objects, keyed by a run name or model identifier. Initialized empty.
    std::map<std::string, model_report*> reports = {}; ///< Map storing model reports (e.g., training progress, evaluation results), keyed by a run name. Initialized empty.
    std::vector<std::thread*> threads = {}; ///< Vector to store pointers to `std::thread` objects for multithreaded execution. Initialized empty.

    std::map<std::string, std::map<std::string, bool>> in_cache = {}; ///< Map to track cache status for data, nested by two string keys (e.g., sample and data type). Initialized empty.
    std::map<std::string, bool> skip_event_build = {}; ///< Map to indicate whether event building should be skipped for certain samples/labels. Initialized empty.
    std::map<std::string, std::string> graph_types = {}; ///< Map to store types of graphs associated with labels. Initialized empty.

    std::vector<folds_t>* tags  = nullptr; ///< Pointer to a vector of `folds_t` (tags for data partitioning/versioning). Initialized to nullptr.
    dataloader*          loader = nullptr; ///< Pointer to the `dataloader` object. Initialized to nullptr.
    sampletracer*        tracer = nullptr; ///< Pointer to the `sampletracer` object. Initialized to nullptr.
    io*                  reader = nullptr; ///< Pointer to the `io` object for data input. Initialized to nullptr.

}; // End of class 'analysis' definition.

#endif // ANALYSIS_H ///< End of include guard for ANALYSIS_H.
