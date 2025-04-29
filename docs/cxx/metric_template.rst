.. cpp:namespace:: metric_template

.. cpp:struct:: metric_t

    Holds the context for a single metric calculation instance.

    This structure encapsulates all the necessary information for executing
    the metric calculation for one specific combination of model checkpoint,
    epoch, k-fold, and device. It provides access to the required input
    variables retrieved from the model and data.

    .. cpp:member:: int kfold = 0
        :brief: The k-fold index (0-based) for this calculation.

    .. cpp:member:: int epoch = 0
        :brief: The epoch number associated with the model checkpoint.

    .. cpp:member:: int device = 0
        :brief: The device index (e.g., GPU ID) where the calculation runs.

    .. cpp:function:: template <typename g> g get(graph_enum grx, std::string name)
        :brief: Retrieves a processed variable by its type and name.
        :tparam g: The expected data type of the variable (e.g., float, std::vector<float>, int).
        :param grx: The graph level/type enum (`graph_enum`) specifying the variable category.
        :param name: The specific name of the variable (e.g., "pt", "eta", "prediction_score").
        :return: The requested variable cast to type `g`. Returns a default-constructed `g`
                 if the variable is not found or if the type cast fails. Prints an error message
                 to stdout on failure.
        :warning: Ensure `build()` has been called before using `get()`.

    .. cpp:function:: std::string mode()
        :brief: Returns the current processing mode as a string.
        :return: std::string "training", "validation", "evaluation", or "undef".

    .. cpp:function:: ~metric_t()
        :brief: Destructor. Cleans up allocated variable_t objects.

        Cleans up resources held by the `metric_t` instance, specifically the `variable_t`
        objects stored in the `handl` map. It iterates through all `variable_t` pointers
        in the `handl` map, sets their `clear` flag to true (if applicable, depending on
        `variable_t`'s design), deletes the object, and sets the pointer to nullptr.

    .. cpp:function:: void build()
        :brief: Builds internal maps (v_maps, h_maps) for fast variable lookup via `get()`.
        :details: This function should be called after the `handl` map (containing pointers to the
                  actual `variable_t` data) has been populated (typically after processing the first
                  graph in `metric_template::execute`).

                  It iterates through the `vars` map (which defines the required variable names).
                  For each required variable name, it finds the corresponding `variable_t` pointer
                  in the `handl` map and stores its index within the vector in the `v_maps` map.
                  It also sets a flag in the `h_maps` map to indicate that this variable is available.
                  These maps (`v_maps`, `h_maps`) allow for faster lookups using `metric_t::get`.
                  Called internally by `metric_template::execute`.

    .. cpp:member:: mode_enum train_mode
        :brief: The current data processing mode (training, validation, etc.).
        :private:

    .. cpp:member:: std::string* pth = nullptr
        :brief: Pointer to the model checkpoint file path string.
        :private:

    .. cpp:member:: model_template* mdlx = nullptr
        :brief: Pointer to the associated model instance.
        :private:

    .. cpp:member:: metric_template* mtx = nullptr
        :brief: Pointer back to the parent metric_template object.
        :private:

    .. cpp:member:: size_t index = 0
        :brief: Unique index assigned to this metric task.
        :private:

    .. cpp:member:: std::map<graph_enum, std::vector<std::string>>* vars = nullptr
        :brief: Pointer to the map defining required variables (owned by metric_template).
        :private:

    .. cpp:member:: std::map<graph_enum, std::vector<variable_t*>>* handl = nullptr
        :brief: Pointer to the map holding the actual processed variable data (populated during execution).
        :private:

    .. cpp:member:: std::map<graph_enum, std::map<std::string, size_t>> v_maps = {}
        :brief: Maps variable enum and name to its index within the `handl` vector for fast access.
        :private:

    .. cpp:member:: std::map<graph_enum, std::map<std::string, bool>> h_maps = {}
        :brief: Maps variable enum and name to a boolean indicating its availability.
        :private:

.. cpp:class:: metric_template : public tools, public notification

    Base class for defining analysis metrics.

    Inherit from this class to implement custom metrics. Provides infrastructure for:
    - Linking to models and run configurations (epochs, k-folds).
    - Requesting specific variables from data and model predictions.
    - Handling data iteration across different modes (train, val, eval).
    - Managing output (e.g., to ROOT files) via the `writer` class.
    - Configuration through properties (`name`, `run_names`, `variables`).

    .. note:: Derived classes MUST implement `define_metric` and SHOULD implement
              `define_variables` if output is needed. `event`, `batch`, and `end`
              provide optional hooks for custom logic at different processing stages.

    .. cpp:function:: metric_template()
        :brief: Default constructor. Initializes properties.
        :details: Initializes the `cproperty` members (`name`, `run_names`, `variables`) by
                  setting their associated object instance (`this`) and their respective
                  getter and setter static methods. This enables the property-like access syntax.

    .. cpp:function:: virtual ~metric_template()
        :brief: Virtual destructor. Cleans up the output writer handle if allocated.
        :details: Performs cleanup. If the `handle` (pointer to a `writer` object, likely for ROOT output)
                  is not null, it deletes the `writer` object and sets the `handle` pointer to nullptr
                  to prevent dangling pointers.

    .. cpp:function:: virtual metric_template* clone()
        :brief: Virtual clone method. Creates a copy of the object.
        :details: Returns a new instance of `metric_template` created using the default constructor.
                  Derived classes should override this to return instances of their own type.
        :return: metric_template* Pointer to the new metric_template instance.
        :note: Derived classes should override this to return an instance of their own type.

    .. cpp:function:: template <typename T> void register_output(std::string tree, std::string name, T* t)
        :brief: Registers an output variable (e.g., a branch in a ROOT TTree).
        :details: Initializes the output writer (`handle`) if it doesn't exist and registers
                  the variable `t` with the specified `tree` and `name` (branch name).
        :tparam T: The data type of the variable to register.
        :param tree: The name of the TTree to associate the variable with.
        :param name: The name of the branch for this variable within the TTree.
        :param t: Pointer to the variable whose address will be associated with the branch.
        :note: Typically called within the `define_variables` method of a derived class.

    .. cpp:function:: template <typename T> void write(std::string tree, std::string name, T* t, bool fill = false)
        :brief: Writes data to a registered output variable and optionally fills the TTree.
        :details: Updates the value of the variable previously registered with `register_output`
                  using the current value pointed to by `t`. If `fill` is true, it also calls
                  the `write` method of the underlying `writer` (likely corresponding to TTree::Fill).
        :tparam T: The data type of the variable.
        :param tree: The name of the TTree containing the variable.
        :param name: The name of the branch (variable).
        :param t: Pointer to the variable containing the data to be written.
        :param fill: If true, triggers the filling of the associated TTree for the current event/entry. Defaults to false.
        :note: Typically called within `define_metric`, `event`, or `batch` methods of a derived class.

    .. cpp:function:: virtual void define_variables()
        :brief: Define output variables/branches using `register_output`. (Optional)
        :details: This method SHOULD be implemented by derived classes if they need to write output.
                  Use `this->register_output<type>("tree_name", "branch_name", &variable)` here.
        :note: Called once before processing starts.

    .. cpp:function:: virtual void define_metric(metric_t* v)
        :brief: Implement the core metric calculation logic using `metric_t::get`. (Mandatory)
        :details: This method MUST be implemented by derived classes to perform the actual metric calculation.
                  Use `v->get<type>("variable_name")` to access required variables.
        :param v: Pointer to the `metric_t` object containing the current context (data, model info).
        :note: Called for each data sample (graph/event). Use `write` here to update output variables.

    .. cpp:function:: virtual void event()
        :brief: Hook called after processing each event/graph. (Optional)
        :details: Derived classes can implement this for event-level aggregation or actions.
        :note: Useful for event-level aggregation.

    .. cpp:function:: virtual void batch()
        :brief: Hook called after processing each data batch. (Optional)
        :details: Derived classes can implement this for batch-level aggregation or actions.
                  Use `this->write<type>("tree_name", "branch_name", &variable, true)` to fill trees.
        :note: Useful for batch-level aggregation or filling output trees (`write` with fill=true).

    .. cpp:function:: virtual void end()
        :brief: Hook called after all data processing is finished. (Optional)
        :details: Derived classes can implement this for final calculations, normalization, or cleanup.
                  Final writing to output files often happens here.
        :note: Useful for final calculations, normalization, saving histograms, or final tree writes.

    .. cpp:member:: cproperty<std::string, metric_template> name
        :brief: Property for setting/getting the name of this metric instance.

    .. cpp:member:: cproperty<std::map<std::string, std::string>, metric_template> run_names
        :brief: Property for setting/getting the map of run identifiers to checkpoint paths.

    .. cpp:member:: cproperty<std::vector<std::string>, metric_template> variables
        :brief: Property for setting/getting the list of required variable strings.

    .. cpp:member:: meta* meta_data = nullptr
        :brief: Pointer to associated metadata object (optional).

    .. cpp:function:: std::map<int, torch::TensorOptions*> get_devices()
        :brief: Retrieves a map of device indices to their corresponding torch::TensorOptions.
        :details: This function iterates through all linked models (`lnks`) associated with this metric instance.
                  For each linked model, it extracts the device index. It ensures that each device index
                  is processed only once. It then stores the `torch::TensorOptions` associated with the
                  model's device in the output map, keyed by the device index.
        :return: std::map<int, torch::TensorOptions*> A map where keys are device indices and
                 values are pointers to the `torch::TensorOptions` for that device.

    .. cpp:function:: std::vector<int> get_kfolds()
        :brief: Retrieves a list of unique k-fold indices used across all epochs and models.
        :details: This function iterates through the `_epoch_kfold` map, which stores checkpoint paths
                  organized by model name, epoch number, and k-fold index. It collects all unique k-fold
                  indices encountered and returns them as a vector.
        :return: std::vector<int> A vector containing the unique k-fold indices used.

    .. cpp:function:: size_t size()
        :brief: Calculates the total number of metric computations to be performed.
        :details: This function iterates through the `_epoch_kfold` map, summing the number of k-folds
                  defined for each epoch across all linked models. This gives the total count of
                  individual metric evaluation instances that will be executed.
        :return: size_t The total number of metric computations (model/epoch/k-fold combinations).

    .. cpp:function:: void construct(std::map<graph_enum, std::vector<variable_t*>>* varx, std::map<graph_enum, std::vector<std::string>>* req, model_template* mdl, graph_t* grx, std::string* mtx)
        :brief: Constructs or updates variable_t objects based on required data from a graph and model.
        :details: This function populates a map of `variable_t` pointers (`varx`) based on a map of required
                  variable names (`req`). It retrieves the necessary data tensors from the provided graph (`grx`)
                  and model (`mdl`).

                  If `varx` is initially smaller than `req`, it resizes `varx` and initializes pointers to nullptr.
                  It then iterates through the required variables (`req`). For each required variable:
                  1. It determines the source of the data (truth, prediction, data) based on the `graph_enum` key.
                  2. It retrieves the corresponding `torch::Tensor` from either the graph (`grx->has_feature`)
                     or the model's prediction maps (`mdl->m_p_graph`, `mdl->m_p_node`, etc.).
                  3. If the tensor is not found, it sets an error message in `mtx` and skips the variable.
                  4. If the tensor is found, it either creates a new `variable_t` object (if the corresponding
                     entry in `varx` is null) or flushes the buffer of the existing one.
                  5. It processes the tensor using the `variable_t::process` method.
                  6. If a new `variable_t` was created (`stx` is true), it updates the `varx` map and sets a
                     success message in `mtx`.
        :param varx: Pointer to a map where keys are `graph_enum` types and values are vectors of
                     pointers to `variable_t` objects. This map will be populated or updated.
        :param req: Pointer to a map specifying the required variables. Keys are `graph_enum` types,
                    and values are vectors of variable names (strings).
        :param mdl: Pointer to the `model_template` instance providing prediction data.
        :param grx: Pointer to the `graph_t` instance providing truth and input data.
        :param mtx: Pointer to a string where status or error messages will be written.

    .. cpp:function:: void execute(metric_t* mtx, metric_template* obj, size_t* prg, std::string* msg)
        :brief: Executes the metric calculation for a specific configuration (epoch, k-fold, model).
        :details: This function orchestrates the metric calculation process for a single `metric_t` instance.
                  1. Initializes the output variable map (`vou`) for the `metric_t` instance.
                  2. Clones the associated model (`mtx->mdlx`) to avoid state conflicts.
                  3. Restores the model's state from the checkpoint specified in `mtx->pth`.
                  4. Creates the output directory path based on epoch, model name, and k-fold.
                  5. Calls `obj->define_variables()` (likely in a derived class) to set up output structures.
                  6. Retrieves the data batches (`hash_bta`) associated with the specific device and k-fold.
                  7. Iterates through the data batches for different modes (training, validation, evaluation).
                  8. For each graph (`gr`) in the batch:
                     a. Performs a forward pass using the model (`mdl->forward`).
                     b. Calls `this->construct` to extract the required variables from the graph and model output.
                     c. If it's the first graph, calls `mtx->build()` to initialize internal mappings within `metric_t`
                        and resets the progress counter.
                     d. Calls `obj->define_metric(mtx)` (likely implemented in a derived class) to perform the
                        actual metric calculation using the extracted variables.
                  9. Calls `obj->end()` (likely in a derived class) for any final processing or cleanup after all batches.
                  10. Cleans up the cloned model and the `metric_t` object.
        :param mtx: Pointer to the `metric_t` structure containing the configuration for this execution.
        :param obj: Pointer to the `metric_template` instance (often `this` or a derived object) that defines the specific metric logic.
        :param prg: Pointer to a size_t used for progress tracking (updated during batch iteration).
        :param msg: Pointer to a string for status messages.

    .. cpp:function:: void define(std::vector<metric_t*>* vr, std::vector<size_t>* num, std::vector<std::string*>* title, size_t* offset)
        :brief: Defines and initializes all `metric_t` instances based on the configuration.
        :details: This function iterates through the `_epoch_kfold` map, which contains the different
                  model/epoch/k-fold combinations for which metrics need to be calculated. For each
                  combination, it creates a new `metric_t` object and populates it with the relevant
                  information:
                  - Checkpoint path (`pth`)
                  - Required variables (`vars`)
                  - Linked model (`mdlx`)
                  - K-fold index (`kfold`)
                  - Epoch number (`epoch`)
                  - Device index (`device`)
                  - Unique index (`index`) within the output vectors
                  - Pointer back to the parent `metric_template` (`mtx`)

                  It also calculates the total number of data samples (`xt`) associated with the
                  specific device and k-fold combination.

                  The created `metric_t` pointers, sample counts, and generated titles are stored
                  in the output vectors `vr`, `num`, and `title` respectively, using the `offset`
                  to determine the correct index.
        :param vr: Pointer to a vector where pointers to the created `metric_t` objects will be stored.
        :param num: Pointer to a vector where the number of samples for each `metric_t` task will be stored.
        :param title: Pointer to a vector where pointers to generated title strings (e.g., "Epoch::X-> K(Y)") will be stored.
        :param offset: Pointer to a size_t used as the index for storing results in the output vectors. It is incremented after each `metric_t` is defined.

    .. cpp:function:: bool link(model_template* mdl)
        :brief: Links a model instance to this metric template.
        :details: This function establishes a connection between a specific `model_template` instance (`mdl`)
                  and this `metric_template`.
                  1. Checks if the model (identified by `mdl->name`) is already linked. If so, returns true.
                  2. Validates if the model name exists as a key in the `_var_type` map (required variables).
                     If not, logs an error and sets `ok` to false.
                  3. Validates if the model name exists as a key in the `_epoch_kfold` map (run configurations).
                     If not, logs an error and sets `ok` to false.
                  4. If validation fails, returns false.
                  5. If validation passes, stores the model pointer in the `lnks` map using the model name as the key.
                  6. Computes hashes based on the model's device index and the k-fold indices associated with
                     this model in `_epoch_kfold`.
                  7. Stores the model pointer in the `hash_mdl` map, keyed by these device/k-fold hashes. This
                     allows efficient lookup of models based on device and k-fold later.
        :param mdl: Pointer to the `model_template` instance to link.
        :return: bool True if the model was linked successfully (including validation checks), false otherwise.

    .. cpp:function:: void link(std::string hsx, std::vector<graph_t*>* data, mode_enum mx)
        :brief: Links a data batch to a specific device/k-fold hash.
        :details: This function associates a vector of data graphs (`data`) with a precomputed hash (`hsx`)
                  representing a specific device and k-fold combination, along with the data's mode
                  (training, validation, or evaluation).

                  It checks if data for this specific hash and mode already exists in `hash_bta`. If not,
                  it stores the pointer to the data vector in the `hash_bta` map.
        :param hsx: The hash string representing the device/k-fold combination.
        :param data: Pointer to a vector of `graph_t*` representing the data batch.
        :param mx: The `mode_enum` value indicating whether the data is for training, validation, or evaluation.

    .. cpp:function:: static void set_name(std::string* name, metric_template* ev)
        :brief: Static setter method for the 'name' property.
        :param name: Pointer to the string containing the new name.
        :param ev: Pointer to the metric_template instance being modified.
        :private:

    .. cpp:function:: static void get_name(std::string* name, metric_template* ev)
        :brief: Static getter method for the 'name' property.
        :param name: Pointer to a string where the current name will be copied.
        :param ev: Pointer to the metric_template instance being accessed.
        :private:

    .. cpp:function:: static void set_run_name(std::map<std::string, std::string>* rn_name, metric_template* ev)
        :brief: Static setter method for the 'run_names' property.
        :details: Parses a map where keys are strings identifying a specific run
                  (e.g., "ModelName::epoch-X::k-Y") and values are the corresponding
                  checkpoint file paths.
                  It validates the format of the key string and the existence of the file path.
                  If valid, it stores the mapping in `ev->_run_names` and also populates
                  the `ev->_epoch_kfold` map for structured access based on model, epoch, and k-fold.
        :param rn_name: Pointer to the map containing run name strings and their file paths.
        :param ev: Pointer to the metric_template instance being modified.
        :private:

    .. cpp:function:: static void get_run_name(std::map<std::string, std::string>* rn_name, metric_template* ev)
        :brief: Static getter method for the 'run_names' property.
        :param rn_name: Pointer to a map where the current run name to file path mappings will be copied.
        :param ev: Pointer to the metric_template instance being accessed.
        :private:

    .. cpp:function:: static void set_variables(std::vector<std::string>* rn_name, metric_template* ev)
        :brief: Static setter method for the 'variables' property.
        :details: Parses a vector of strings, each specifying a required variable using the format:
                  "<ModelName>::<Level>::<Type>::<variable>"
                    - Level: data, truth, prediction
                    - Type: edge, node, graph, extra
                    - variable: Specific name (e.g., pt, eta, index, weight)

                  It validates the format and determines the appropriate `graph_enum` based on the
                  Level, Type, and variable name. The variable name is then stored under the
                  corresponding model name and `graph_enum` type in the `ev->_var_type` map.
                  The original string is stored in `ev->_variables` for tracking.
        :param rn_name: Pointer to a vector of strings specifying the required variables.
        :param ev: Pointer to the metric_template instance being modified.
        :private:

    .. cpp:function:: static void get_variables(std::vector<std::string>* rn_name, metric_template* ev)
        :brief: Static getter method for the 'variables' property.
        :param rn_name: Pointer to a vector where the original variable specification strings will be added.
        :param ev: Pointer to the metric_template instance being accessed.
        :private:

    .. cpp:function:: metric_template* clone(int)
        :brief: Internal clone method used for creating execution copies.
        :details: Creates a new instance using `clone()` and then copies the essential configuration maps
                  (`_var_type`, `_epoch_kfold`) from the current object to the new clone. This allows
                  different threads or processes to have their own copy of the metric configuration
                  while potentially sharing underlying model or data structures (not copied here).
        :param Unused integer parameter (likely a placeholder or legacy).
        :return: metric_template* A pointer to a new `metric_template` object with copied configuration.
        :private:

    .. cpp:member:: std::map<std::string, model_template*> lnks
        :brief: Map linking model names to their corresponding model_template instances.
        :private:

    .. cpp:member:: std::map<std::string, std::vector<model_template*>> hash_mdl = {}
        :brief: Map linking device/k-fold hashes to the models associated with them.
        :private:

    .. cpp:member:: std::map<std::string, std::map<mode_enum, std::vector<graph_t*>*>> hash_bta = {}
        :brief: Map linking device/k-fold hashes to data batches for different modes (train, val, eval).
        :private:

    .. cpp:member:: std::map<std::string, std::map<int, std::map<int, std::string>>> _epoch_kfold
        :brief: Stores checkpoint paths, structured by [ModelName][Epoch][KFoldIndex].
        :private:

    .. cpp:member:: std::map<std::string, std::map<graph_enum, std::vector<std::string>>> _var_type
        :brief: Stores required variable names, structured by [ModelName][graph_enum].
        :private:

    .. cpp:member:: std::string _name = "metric-template"
        :brief: The name assigned to this metric instance.
        :private:

    .. cpp:member:: std::string _outdir = ""
        :brief: Base output directory path (modified during execution).
        :private:

    .. cpp:member:: std::map<std::string, std::string> _run_names = {}
        :brief: Internal storage for the 'run_names' property map.
        :private:

    .. cpp:member:: std::map<std::string, std::string> _variables = {}
        :brief: Internal storage for the 'variables' property (stores original specification strings).
        :private:

    .. cpp:member:: writer* handle = nullptr
        :brief: Handle for the output writer (e.g., ROOT file writer).
        :private:

.. cpp:function:: std::string enums_to_string(graph_enum gr)
    :brief: Converts a graph_enum value to its corresponding string representation.
    :details: This utility function takes a `graph_enum` value (representing different types or levels
              of graph data like truth, prediction, node, edge, etc.) and returns a predefined string
              prefix associated with it. This is useful for constructing unique names or identifiers.
    :param gr: The graph_enum value to convert.
    :return: std::string The string representation of the enum value, or "undef" if not recognized.

.. file:: metric_template.dox
    :brief: Implementation details for the metric_template class and related structures.

    This file provides the implementation for methods associated with the metric_template
    class, which serves as a base class for defining custom metrics in the analysis framework.
    It includes methods for managing device information, k-folds, variable construction,
    metric execution, and linking with models and data.
/**
 * @file metric_template.dox
 * @brief Implementation details for the metric_template class and related structures.
 *
 * This file provides the implementation for methods associated with the metric_template
 * class, which serves as a base class for defining custom metrics in the analysis framework.
 * It includes methods for managing device information, k-folds, variable construction,
 * metric execution, and linking with models and data.
 */

#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <map>
#include <vector>
#include <string>
#include <torch/torch.h> // Assuming torch::TensorOptions is from libtorch
#include <chrono>        // For std::chrono
#include <thread>        // For std::this_thread

// Forward declarations if needed
struct graph_t;
class model_template;
struct variable_t; // Assuming variable_t is defined elsewhere
enum class graph_enum; // Assuming graph_enum is defined elsewhere
enum class mode_enum; // Assuming mode_enum is defined elsewhere
struct metric_t; // Assuming metric_t is defined elsewhere

/**
 * @brief Retrieves a map of device indices to their corresponding torch::TensorOptions.
 *
 * This function iterates through all linked models (`lnks`) associated with this metric instance.
 * For each linked model, it extracts the device index. It ensures that each device index
 * is processed only once. It then stores the `torch::TensorOptions` associated with the
 * model's device in the output map, keyed by the device index.
 *
 * @return std::map<int, torch::TensorOptions*> A map where keys are device indices and
 *         values are pointers to the `torch::TensorOptions` for that device.
 */
std::map<int, torch::TensorOptions*> metric_template::get_devices();

/**
 * @brief Retrieves a list of unique k-fold indices used across all epochs and models.
 *
 * This function iterates through the `_epoch_kfold` map, which stores checkpoint paths
 * organized by model name, epoch number, and k-fold index. It collects all unique k-fold
 * indices encountered and returns them as a vector.
 *
 * @return std::vector<int> A vector containing the unique k-fold indices used.
 */
std::vector<int> metric_template::get_kfolds();

/**
 * @brief Calculates the total number of metric computations to be performed.
 *
 * This function iterates through the `_epoch_kfold` map, summing the number of k-folds
 * defined for each epoch across all linked models. This gives the total count of
 * individual metric evaluation instances that will be executed.
 *
 * @return size_t The total number of metric computations (model/epoch/k-fold combinations).
 */
size_t metric_template::size();

/**
 * @brief Converts a graph_enum value to its corresponding string representation.
 *
 * This utility function takes a `graph_enum` value (representing different types or levels
 * of graph data like truth, prediction, node, edge, etc.) and returns a predefined string
 * prefix associated with it. This is useful for constructing unique names or identifiers.
 *
 * @param gr The graph_enum value to convert.
 * @return std::string The string representation of the enum value, or "undef" if not recognized.
 */
std::string enums_to_string(graph_enum gr);

/**
 * @brief Constructs or updates variable_t objects based on required data from a graph and model.
 *
 * This function populates a map of `variable_t` pointers (`varx`) based on a map of required
 * variable names (`req`). It retrieves the necessary data tensors from the provided graph (`grx`)
 * and model (`mdl`).
 *
 * If `varx` is initially smaller than `req`, it resizes `varx` and initializes pointers to nullptr.
 * It then iterates through the required variables (`req`). For each required variable:
 * 1. It determines the source of the data (truth, prediction, data) based on the `graph_enum` key.
 * 2. It retrieves the corresponding `torch::Tensor` from either the graph (`grx->has_feature`)
 *    or the model's prediction maps (`mdl->m_p_graph`, `mdl->m_p_node`, etc.).
 * 3. If the tensor is not found, it sets an error message in `mtx` and skips the variable.
 * 4. If the tensor is found, it either creates a new `variable_t` object (if the corresponding
 *    entry in `varx` is null) or flushes the buffer of the existing one.
 * 5. It processes the tensor using the `variable_t::process` method.
 * 6. If a new `variable_t` was created (`stx` is true), it updates the `varx` map and sets a
 *    success message in `mtx`.
 *
 * @param varx Pointer to a map where keys are `graph_enum` types and values are vectors of
 *             pointers to `variable_t` objects. This map will be populated or updated.
 * @param req Pointer to a map specifying the required variables. Keys are `graph_enum` types,
 *            and values are vectors of variable names (strings).
 * @param mdl Pointer to the `model_template` instance providing prediction data.
 * @param grx Pointer to the `graph_t` instance providing truth and input data.
 * @param mtx Pointer to a string where status or error messages will be written.
 */
void metric_template::construct(
    std::map<graph_enum, std::vector<variable_t*>>* varx,
    std::map<graph_enum, std::vector<std::string>>* req,
    model_template* mdl, graph_t* grx, std::string* mtx
);

/**
 * @brief Executes the metric calculation for a specific configuration (epoch, k-fold, model).
 *
 * This function orchestrates the metric calculation process for a single `metric_t` instance.
 * 1. Initializes the output variable map (`vou`) for the `metric_t` instance.
 * 2. Clones the associated model (`mtx->mdlx`) to avoid state conflicts.
 * 3. Restores the model's state from the checkpoint specified in `mtx->pth`.
 * 4. Creates the output directory path based on epoch, model name, and k-fold.
 * 5. Calls `obj->define_variables()` (likely in a derived class) to set up output structures.
 * 6. Retrieves the data batches (`hash_bta`) associated with the specific device and k-fold.
 * 7. Iterates through the data batches for different modes (training, validation, evaluation).
 * 8. For each graph (`gr`) in the batch:
 *    a. Performs a forward pass using the model (`mdl->forward`).
 *    b. Calls `this->construct` to extract the required variables from the graph and model output.
 *    c. If it's the first graph, calls `mtx->build()` to initialize internal mappings within `metric_t`
 *       and resets the progress counter.
 *    d. Calls `obj->define_metric(mtx)` (likely implemented in a derived class) to perform the
 *       actual metric calculation using the extracted variables.
 * 9. Calls `obj->end()` (likely in a derived class) for any final processing or cleanup after all batches.
 * 10. Cleans up the cloned model and the `metric_t` object.
 *
 * @param mtx Pointer to the `metric_t` structure containing the configuration for this execution.
 * @param obj Pointer to the `metric_template` instance (often `this` or a derived object) that defines the specific metric logic.
 * @param prg Pointer to a size_t used for progress tracking (updated during batch iteration).
 * @param msg Pointer to a string for status messages.
 */
void metric_template::execute(metric_t* mtx, metric_template* obj, size_t* prg, std::string* msg);

/**
 * @brief Defines and initializes all `metric_t` instances based on the configuration.
 *
 * This function iterates through the `_epoch_kfold` map, which contains the different
 * model/epoch/k-fold combinations for which metrics need to be calculated. For each
 * combination, it creates a new `metric_t` object and populates it with the relevant
 * information:
 * - Checkpoint path (`pth`)
 * - Required variables (`vars`)
 * - Linked model (`mdlx`)
 * - K-fold index (`kfold`)
 * - Epoch number (`epoch`)
 * - Device index (`device`)
 * - Unique index (`index`) within the output vectors
 * - Pointer back to the parent `metric_template` (`mtx`)
 *
 * It also calculates the total number of data samples (`xt`) associated with the
 * specific device and k-fold combination.
 *
 * The created `metric_t` pointers, sample counts, and generated titles are stored
 * in the output vectors `vr`, `num`, and `title` respectively, using the `offset`
 * to determine the correct index.
 *
 * @param vr Pointer to a vector where pointers to the created `metric_t` objects will be stored.
 * @param num Pointer to a vector where the number of samples for each `metric_t` task will be stored.
 * @param title Pointer to a vector where pointers to generated title strings (e.g., "Epoch::X-> K(Y)") will be stored.
 * @param offset Pointer to a size_t used as the index for storing results in the output vectors. It is incremented after each `metric_t` is defined.
 */
void metric_template::define(std::vector<metric_t*>* vr, std::vector<size_t>* num, std::vector<std::string*>* title, size_t* offset);

// --- Implementation for linking models and data ---

#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <vector>
#include <string>
#include <map>

/**
 * @brief Links a model instance to this metric template.
 *
 * This function establishes a connection between a specific `model_template` instance (`mdl`)
 * and this `metric_template`.
 * 1. Checks if the model (identified by `mdl->name`) is already linked. If so, returns true.
 * 2. Validates if the model name exists as a key in the `_var_type` map (required variables).
 *    If not, logs an error and sets `ok` to false.
 * 3. Validates if the model name exists as a key in the `_epoch_kfold` map (run configurations).
 *    If not, logs an error and sets `ok` to false.
 * 4. If validation fails, returns false.
 * 5. If validation passes, stores the model pointer in the `lnks` map using the model name as the key.
 * 6. Computes hashes based on the model's device index and the k-fold indices associated with
 *    this model in `_epoch_kfold`.
 * 7. Stores the model pointer in the `hash_mdl` map, keyed by these device/k-fold hashes. This
 *    allows efficient lookup of models based on device and k-fold later.
 *
 * @param mdl Pointer to the `model_template` instance to link.
 * @return bool True if the model was linked successfully (including validation checks), false otherwise.
 */
bool metric_template::link(model_template* mdl);

/**
 * @brief Links a data batch to a specific device/k-fold hash.
 *
 * This function associates a vector of data graphs (`data`) with a precomputed hash (`hsx`)
 * representing a specific device and k-fold combination, along with the data's mode
 * (training, validation, or evaluation).
 *
 * It checks if data for this specific hash and mode already exists in `hash_bta`. If not,
 * it stores the pointer to the data vector in the `hash_bta` map.
 *
 * @param hsx The hash string representing the device/k-fold combination.
 * @param data Pointer to a vector of `graph_t*` representing the data batch.
 * @param mx The `mode_enum` value indicating whether the data is for training, validation, or evaluation.
 */
void metric_template::link(std::string hsx, std::vector<graph_t*>* data, mode_enum mx);


// --- Implementation for metric_t struct ---

#include <templates/metric_template.h> // Contains definition of metric_t, variable_t, graph_enum
#include <iostream> // For error messages
#include <map>
#include <vector>
#include <string>

/**
 * @brief Destructor for the metric_t struct.
 *
 * Cleans up resources held by the `metric_t` instance, specifically the `variable_t`
 * objects stored in the `handl` map. It iterates through all `variable_t` pointers
 * in the `handl` map, sets their `clear` flag to true (if applicable, depending on
 * `variable_t`'s design), deletes the object, and sets the pointer to nullptr.
 */
metric_t::~metric_t();

/**
 * @brief Builds internal mappings for quick variable access.
 *
 * This function should be called after the `handl` map (containing pointers to the
 * actual `variable_t` data) has been populated (typically after processing the first
 * graph in `metric_template::execute`).
 *
 * It iterates through the `vars` map (which defines the required variable names).
 * For each required variable name, it finds the corresponding `variable_t` pointer
 * in the `handl` map and stores its index within the vector in the `v_maps` map.
 * It also sets a flag in the `h_maps` map to indicate that this variable is available.
 * These maps (`v_maps`, `h_maps`) allow for faster lookups using `metric_t::get`.
 */
void metric_t::build();

/**
 * @brief Returns the string representation of the current processing mode.
 *
 * Converts the internal `train_mode` (which is of type `mode_enum`) into a
 * human-readable string ("training", "validation", "evaluation", or "undef").
 *
 * @return std::string The string representation of the current mode.
 */
std::string metric_t::mode();


// --- Implementation for metric_template constructor, destructor, and virtual methods ---

#include <templates/metric_template.h>
#include <templates/model_template.h> // Included for potential use, though not directly in this snippet
#include <meta/meta.h> // Included for meta* meta_data member

/**
 * @brief Constructor for the metric_template class.
 *
 * Initializes the `cproperty` members (`name`, `run_names`, `variables`) by
 * setting their associated object instance (`this`) and their respective
 * getter and setter static methods. This enables the property-like access syntax.
 */
metric_template::metric_template();

/**
 * @brief Destructor for the metric_template class.
 *
 * Performs cleanup. If the `handle` (pointer to a `writer` object, likely for ROOT output)
 * is not null, it deletes the `writer` object and sets the `handle` pointer to nullptr
 * to prevent dangling pointers.
 */
metric_template::~metric_template();

/**
 * @brief Creates a basic clone of the metric_template object.
 *
 * Returns a new instance of `metric_template` created using the default constructor.
 * Derived classes should override this to return instances of their own type.
 *
 * @return metric_template* A pointer to a new `metric_template` object.
 */
metric_template* metric_template::clone();

/**
 * @brief Creates a clone suitable for parallel execution (internal use).
 *
 * Creates a new instance using `clone()` and then copies the essential configuration maps
 * (`_var_type`, `_epoch_kfold`) from the current object to the new clone. This allows
 * different threads or processes to have their own copy of the metric configuration
 * while potentially sharing underlying model or data structures (not copied here).
 *
 * @param Unused integer parameter (likely a placeholder or legacy).
 * @return metric_template* A pointer to a new `metric_template` object with copied configuration.
 */
metric_template* metric_template::clone(int);

/**
 * @brief Virtual method to define the core metric calculation logic.
 * @param v Pointer to the `metric_t` object containing the current context (data, model info).
 *          Use `v->get<type>("variable_name")` to access required variables.
 * @note This method MUST be implemented by derived classes to perform the actual metric calculation.
 */
void metric_template::define_metric(metric_t* v);

/**
 * @brief Virtual method to define output variables (e.g., ROOT tree branches).
 * @note This method SHOULD be implemented by derived classes if they need to write output.
 *       Use `this->register_output<type>("tree_name", "branch_name", &variable)` here.
 */
void metric_template::define_variables();

/**
 * @brief Virtual method called potentially after processing each event/graph (if applicable).
 * @note Derived classes can implement this for event-level aggregation or actions.
 */
void metric_template::event();

/**
 * @brief Virtual method called potentially after processing each batch of data.
 * @note Derived classes can implement this for batch-level aggregation or actions.
 *       Use `this->write<type>("tree_name", "branch_name", &variable, true)` to fill trees.
 */
void metric_template::batch();

/**
 * @brief Virtual method called after all processing is complete.
 * @note Derived classes can implement this for final calculations, normalization, or cleanup.
 *       Final writing to output files often happens here.
 */
void metric_template::end();


// --- Implementation for metric_template property setters/getters ---

#include <templates/metric_template.h>
#include <vector>
#include <string>
#include <map>
#include <stdexcept> // Potentially for std::stoi errors

/**
 * @brief Static setter method for the 'name' property.
 * @param name Pointer to the string containing the new name.
 * @param ev Pointer to the metric_template instance being modified.
 */
void metric_template::set_name(std::string* name, metric_template* ev);

/**
 * @brief Static getter method for the 'name' property.
 * @param name Pointer to a string where the current name will be copied.
 * @param ev Pointer to the metric_template instance being accessed.
 */
void metric_template::get_name(std::string* name, metric_template* ev);

/**
 * @brief Static setter method for the 'run_names' property.
 *
 * Parses a map where keys are strings identifying a specific run
 * (e.g., "ModelName::epoch-X::k-Y") and values are the corresponding
 * checkpoint file paths.
 * It validates the format of the key string and the existence of the file path.
 * If valid, it stores the mapping in `ev->_run_names` and also populates
 * the `ev->_epoch_kfold` map for structured access based on model, epoch, and k-fold.
 *
 * @param rn_name Pointer to the map containing run name strings and their file paths.
 * @param ev Pointer to the metric_template instance being modified.
 */
void metric_template::set_run_name(std::map<std::string, std::string>* rn_name, metric_template* ev);

/**
 * @brief Static getter method for the 'run_names' property.
 * @param rn_name Pointer to a map where the current run name to file path mappings will be copied.
 * @param ev Pointer to the metric_template instance being accessed.
 */
void metric_template::get_run_name(std::map<std::string, std::string>* rn_name, metric_template* ev);

/**
 * @brief Static setter method for the 'variables' property.
 *
 * Parses a vector of strings, each specifying a required variable using the format:
 * "<ModelName>::<Level>::<Type>::<variable>"
 *   - Level: data, truth, prediction
 *   - Type: edge, node, graph, extra
 *   - variable: Specific name (e.g., pt, eta, index, weight)
 *
 * It validates the format and determines the appropriate `graph_enum` based on the
 * Level, Type, and variable name. The variable name is then stored under the
 * corresponding model name and `graph_enum` type in the `ev->_var_type` map.
 * The original string is stored in `ev->_variables` for tracking.
 *
 * @param rn_name Pointer to a vector of strings specifying the required variables.
 * @param ev Pointer to the metric_template instance being modified.
 */
void metric_template::set_variables(std::vector<std::string>* rn_name, metric_template* ev);

/**
 * @brief Static getter method for the 'variables' property.
 * @param rn_name Pointer to a vector where the original variable specification strings will be added.
 * @param ev Pointer to the metric_template instance being accessed.
 */
void metric_template::get_variables(std::vector<std::string>* rn_name, metric_template* ev);


// --- Header file content for metric_template.h ---

#ifndef METRIC_TEMPLATE_H
#define METRIC_TEMPLATE_H

#include <notification/notification.h> // Base class for logging/messaging
#include <structs/property.h>        // For cproperty template
#include <structs/element.h>         // Likely contains variable_t definition
#include <structs/event.h>           // Potentially related to graph_t or data structure
#include <structs/model.h>           // Potentially related to model_template
#include <structs/enums.h>           // Contains graph_enum, mode_enum
#include <meta/meta.h>               // Contains meta class definition

#include <plotting/plotting.h>       // For writer class (ROOT output)
#include <tools/vector_cast.h>       // Utility for casting vectors
#include <tools/merge_cast.h>        // Utility for merging/casting
#include <tools/tools.h>             // Base class providing utility functions (split, is_file, etc.)

#include <map>
#include <vector>
#include <string>
#include <iostream> // For metric_t::get error message
#include <torch/torch.h> // For torch::TensorOptions

// Forward declarations
struct graph_t;         // Represents a single graph/event data structure
class analysis;         // Main analysis class, likely orchestrates metrics
class model_template;   // Base class for models
class metric_template;  // Forward declare the main class
class writer;           // ROOT output writer class

/**
 * @struct metric_t
 * @brief Holds the context for a single metric calculation instance.
 *
 * This structure encapsulates all the necessary information for executing
 * the metric calculation for one specific combination of model checkpoint,
 * epoch, k-fold, and device. It provides access to the required input
 * variables retrieved from the model and data.
 */
struct metric_t {
    public:
    /**
     * @brief Destructor. Cleans up allocated variable_t objects.
     */
    ~metric_t();

    int kfold = 0;  ///< The k-fold index (0-based) for this calculation.
    int epoch = 0;  ///< The epoch number associated with the model checkpoint.
    int device = 0; ///< The device index (e.g., GPU ID) where the calculation runs.

    /**
     * @brief Retrieves a processed variable by its type and name.
     * @tparam g The expected data type of the variable (e.g., float, std::vector<float>, int).
     * @param grx The graph level/type enum (`graph_enum`) specifying the variable category.
     * @param name The specific name of the variable (e.g., "pt", "eta", "prediction_score").
     * @return g The requested variable cast to type `g`. Returns a default-constructed `g`
     *           if the variable is not found or if the type cast fails. Prints an error message
     *           to stdout on failure.
     * @warning Ensure `build()` has been called before using `get()`.
     */
    template <typename g>
    g get(graph_enum grx, std::string name);

    /**
     * @brief Returns the current processing mode as a string.
     * @return std::string "training", "validation", "evaluation", or "undef".
     */
    std::string mode();

    private:
    // Allow metric_template and analysis to access private members.
    friend metric_template;
    friend analysis;

    /**
     * @brief Builds internal maps (v_maps, h_maps) for fast variable lookup via `get()`.
     * Called internally by `metric_template::execute`.
     */
    void build();

    mode_enum train_mode; ///< The current data processing mode (training, validation, etc.).
    std::string* pth = nullptr; ///< Pointer to the model checkpoint file path string.
    model_template* mdlx = nullptr; ///< Pointer to the associated model instance.
    metric_template* mtx = nullptr; ///< Pointer back to the parent metric_template object.
    size_t index = 0; ///< Unique index assigned to this metric task.

    /// Pointer to the map defining required variables (owned by metric_template).
    std::map<graph_enum, std::vector<std::string>>* vars = nullptr;
    /// Pointer to the map holding the actual processed variable data (populated during execution).
    std::map<graph_enum, std::vector<variable_t*>>* handl = nullptr;
    /// Maps variable enum and name to its index within the `handl` vector for fast access.
    std::map<graph_enum, std::map<std::string, size_t>> v_maps = {};
    /// Maps variable enum and name to a boolean indicating its availability.
    std::map<graph_enum, std::map<std::string, bool>>   h_maps = {};
};


/**
 * @class metric_template
 * @brief Base class for defining analysis metrics.
 *
 * Inherit from this class to implement custom metrics. Provides infrastructure for:
 * - Linking to models and run configurations (epochs, k-folds).
 * - Requesting specific variables from data and model predictions.
 * - Handling data iteration across different modes (train, val, eval).
 * - Managing output (e.g., to ROOT files) via the `writer` class.
 * - Configuration through properties (`name`, `run_names`, `variables`).
 *
 * @note Derived classes MUST implement `define_metric` and SHOULD implement
 *       `define_variables` if output is needed. `event`, `batch`, and `end`
 *       provide optional hooks for custom logic at different processing stages.
 */
class metric_template:
    public tools,        // Provides utility functions (string manipulation, file checks)
    public notification  // Provides logging methods (info, warning, failure, success)
{
    public:
    /**
     * @brief Default constructor. Initializes properties.
     */
    metric_template();

    /**
     * @brief Virtual destructor. Cleans up the output writer handle if allocated.
     */
    virtual ~metric_template();

    /**
     * @brief Virtual clone method. Creates a copy of the object.
     * @note Derived classes should override this to return an instance of their own type.
     * @return metric_template* Pointer to the new metric_template instance.
     */
    virtual metric_template* clone();

    /**
     * @brief Registers an output variable (e.g., a branch in a ROOT TTree).
     *
     * Initializes the output writer (`handle`) if it doesn't exist and registers
     * the variable `t` with the specified `tree` and `name` (branch name).
     *
     * @tparam T The data type of the variable to register.
     * @param tree The name of the TTree to associate the variable with.
     * @param name The name of the branch for this variable within the TTree.
     * @param t Pointer to the variable whose address will be associated with the branch.
     * @note Typically called within the `define_variables` method of a derived class.
     */
    template <typename T>
    void register_output(std::string tree, std::string name, T* t);

    /**
     * @brief Writes data to a registered output variable and optionally fills the TTree.
     *
     * Updates the value of the variable previously registered with `register_output`
     * using the current value pointed to by `t`. If `fill` is true, it also calls
     * the `write` method of the underlying `writer` (likely corresponding to TTree::Fill).
     *
     * @tparam T The data type of the variable.
     * @param tree The name of the TTree containing the variable.
     * @param name The name of the branch (variable).
     * @param t Pointer to the variable containing the data to be written.
     * @param fill If true, triggers the filling of the associated TTree for the current event/entry. Defaults to false.
     * @note Typically called within `define_metric`, `event`, or `batch` methods of a derived class.
     */
    template <typename T>
    void write(std::string tree, std::string name, T* t, bool fill = false);

    // --- Virtual methods for derived classes to implement ---

    /**
     * @brief Define output variables/branches using `register_output`. (Optional)
     * @note Called once before processing starts.
     */
    virtual void define_variables();

    /**
     * @brief Implement the core metric calculation logic using `metric_t::get`. (Mandatory)
     * @param v Pointer to the `metric_t` context for the current calculation step.
     * @note Called for each data sample (graph/event). Use `write` here to update output variables.
     */
    virtual void define_metric(metric_t* v);

    /**
     * @brief Hook called after processing each event/graph. (Optional)
     * @note Useful for event-level aggregation.
     */
    virtual void event();

    /**
     * @brief Hook called after processing each data batch. (Optional)
     * @note Useful for batch-level aggregation or filling output trees (`write` with fill=true).
     */
    virtual void batch();

    /**
     * @brief Hook called after all data processing is finished. (Optional)
     * @note Useful for final calculations, normalization, saving histograms, or final tree writes.
     */
    virtual void end();

    // --- Configurable Properties ---

    /// Property for setting/getting the name of this metric instance.
    cproperty<std::string, metric_template> name;
    /// Property for setting/getting the map of run identifiers to checkpoint paths.
    cproperty<std::map<std::string, std::string>, metric_template> run_names;
    /// Property for setting/getting the list of required variable strings.
    cproperty<std::vector<std::string>, metric_template> variables;
    /// Pointer to associated metadata object (optional).
    meta* meta_data = nullptr;

    private:
    // Allow analysis class to access private members/methods for orchestration.
    friend analysis;

    // --- Internal Data Structures ---

    /// Map linking model names to their corresponding model_template instances.
    std::map<std::string, model_template*> lnks;
    /// Map linking device/k-fold hashes to the models associated with them.
    std::map<std::string, std::vector<model_template*>> hash_mdl = {};
    /// Map linking device/k-fold hashes to data batches for different modes (train, val, eval).
    std::map<std::string, std::map<mode_enum, std::vector<graph_t*>*>> hash_bta = {};
    /// Stores checkpoint paths, structured by [ModelName][Epoch][KFoldIndex].
    std::map<std::string, std::map<int, std::map<int, std::string>>> _epoch_kfold;
    /// Stores required variable names, structured by [ModelName][graph_enum].
    std::map<std::string, std::map<graph_enum, std::vector<std::string>>> _var_type;

    // --- Internal State ---

    /// The name assigned to this metric instance.
    std::string _name = "metric-template";
    /// Base output directory path (modified during execution).
    std::string _outdir = "";
    /// Internal storage for the 'run_names' property map.
    std::map<std::string, std::string> _run_names = {};
    /// Internal storage for the 'variables' property (stores original specification strings).
    std::map<std::string, std::string> _variables = {};

    // --- Static Property Accessors ---

    /// Static setter for the 'name' property.
    void static set_name(std::string*, metric_template*);
    /// Static getter for the 'name' property.
    void static get_name(std::string*, metric_template*);

    /// Static setter for the 'run_names' property.
    void static set_run_name(std::map<std::string, std::string>*, metric_template*);
    /// Static getter for the 'run_names' property.
    void static get_run_name(std::map<std::string, std::string>*, metric_template*);

    /// Static setter for the 'variables' property.
    void static set_variables(std::vector<std::string>*, metric_template*);
    /// Static getter for the 'variables' property.
    void static get_variables(std::vector<std::string>*, metric_template*);

    // --- Internal Helper Methods ---

    /**
     * @brief Constructs variable_t objects from graph and model data. (Static helper)
     * @param varx Output map for variable_t pointers.
     * @param req Input map specifying required variables.
     * @param mdl Model providing prediction data.
     * @param grx Graph providing truth/input data.
     * @param mtx String for status/error messages.
     */
    void static construct(
        std::map<graph_enum, std::vector<variable_t*>>* varx,
        std::map<graph_enum, std::vector<std::string>>* req,
        model_template* mdl, graph_t* grx, std::string* mtx
    );

    /**
     * @brief Internal clone method used for creating execution copies.
     * @param Unused integer parameter.
     * @return metric_template* Pointer to the cloned instance with copied configurations.
     */
    metric_template* clone(int);

    /**
     * @brief Links a model instance to this metric.
     * @param mdl Pointer to the model to link.
     * @return bool True on success, false on validation failure.
     */
    bool link(model_template*);

    /**
     * @brief Links a data batch to a device/k-fold hash and mode.
     * @param hsx The device/k-fold hash string.
     * @param data Pointer to the vector of graph data.
     * @param mx The mode (training, validation, evaluation).
     */
    void link(std::string hsx, std::vector<graph_t*>* data, mode_enum mx);

    /**
     * @brief Executes the metric calculation for a single metric_t configuration.
     * @param mtx The metric context.
     * @param obj The metric object (usually derived instance).
     * @param prg Progress counter.
     * @param msg Status message string.
     */
    void execute(metric_t* mtx, metric_template* obj, size_t* prg, std::string* msg);

    /**
     * @brief Defines all metric_t tasks based on configuration.
     * @param vr Output vector for metric_t pointers.
     * @param num Output vector for sample counts per task.
     * @param title Output vector for task title strings.
     * @param offset Input/Output offset index for populating vectors.
     */
    void define(
        std::vector<metric_t*>* vr, std::vector<size_t>* num,
        std::vector<std::string*>* title, size_t* offset
    );

    /**
     * @brief Calculates the total number of metric tasks.
     * @return size_t The total count of model/epoch/k-fold combinations.
     */
    size_t size();

    /**
     * @brief Gets the unique device options used by linked models.
     * @return std::map<int, torch::TensorOptions*> Map of device index to tensor options.
     */
    std::map<int, torch::TensorOptions*> get_devices();

    /**
     * @brief Gets the unique k-fold indices used across all configurations.
     * @return std::vector<int> Vector of unique k-fold indices.
     */
    std::vector<int> get_kfolds();

    /// Handle for the output writer (e.g., ROOT file writer).
    writer* handle = nullptr;
};


#endif // METRIC_TEMPLATE_H