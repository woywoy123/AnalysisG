.. cpp:namespace:: detail

.. comment::
    This file defines structures and functions for data handling,
    including dictionary generation, data type translation,
    and interfacing with ROOT trees.

.. cpp:function:: void buildDict(std::string _name, std::string _shrt)

   Builds a dictionary for a given data type.

   This function uses the ROOT interpreter to generate a dictionary
   for the specified data type, allowing it to be used in ROOT
   macros and applications.

   :param _name: The name of the data type (e.g., "vector<double>").
   :param _shrt: The include path for the data type (e.g., "vector").

.. cpp:function:: void registerInclude(std::string pth, bool is_abs)

   Registers an include path with the ROOT interpreter.

   This function adds an include path to the ROOT interpreter's
   search path, allowing it to find header files for data types
   used in the analysis.

   :param pth: The include path to register.
   :param is_abs: A boolean indicating whether the path is absolute (true) or relative (false).

.. cpp:function:: void buildPCM(std::string name, std::string incl, bool exl)

   Builds a dictionary and optionally excludes it.

   This function combines the dictionary building with an exclusion
   option. If ``exl`` is true, the dictionary is not built.

   :param name: The name of the data type.
   :param incl: The include path for the data type.
   :param exl: A boolean indicating whether to exclude the data type from dictionary generation.

.. cpp:function:: void buildAll()

   Builds dictionaries for all supported data types.

   This function calls ``buildPCM`` for each supported data type,
   ensuring that dictionaries are available for all commonly used
   types in the analysis.

.. cpp:struct:: bsc_t

   .. cpp:member:: bsc_t()

      Default constructor for the ``bsc_t`` struct.

   .. cpp:member:: ~bsc_t()

      Virtual destructor for the ``bsc_t`` struct.

   .. cpp:member:: data_enum root_type_translate(std::string* root_str)

      Translates a ROOT type string to a ``data_enum`` value.

      This function takes a string representing a ROOT data type and
      returns the corresponding ``data_enum`` value. It handles both
      primitive types and vectors of various depths.

      :param root_str: A pointer to the ROOT type string.
      :return: The corresponding ``data_enum`` value.

   .. cpp:member:: std::string as_string()

      Returns the string representation of the ``data_enum`` type.

      This function returns a string representation of the ``data_enum``
      type stored in the ``bsc_t`` object.

      :return: The string representation of the ``data_enum`` type.

   .. cpp:member:: std::string scan_buffer()

      Scans the buffer and returns a string of set data types.

      This function checks which data type flags are set in the ``bsc_t``
      object and returns a string containing a list of the set types.

      :return: A string listing the set data types.

   .. cpp:member:: void flush_buffer()

      Flushes the data buffer based on the current data type.

      This function clears the data buffer associated with the current
      ``data_enum`` type in the ``bsc_t`` object.

   .. cpp:member:: bool element(std::vector<std::vector<std::vector<float>>>* el)

      Sets the element value for a vector of vector of vector of floats.

      :param el: Pointer to the vector of vector of vector of floats.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<std::vector<double>>>* el)

      Sets the element value for a vector of vector of vector of doubles.

      :param el: Pointer to the vector of vector of vector of doubles.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<std::vector<long>>>* el)

      Sets the element value for a vector of vector of vector of longs.

      :param el: Pointer to the vector of vector of vector of longs.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<std::vector<int>>>* el)

      Sets the element value for a vector of vector of vector of ints.

      :param el: Pointer to the vector of vector of vector of ints.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<std::vector<bool>>>* el)

      Sets the element value for a vector of vector of vector of bools.

      :param el: Pointer to the vector of vector of vector of bools.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<float>>* el)

      Sets the element value for a vector of vector of floats.

      :param el: Pointer to the vector of vector of floats.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<double>>* el)

      Sets the element value for a vector of vector of doubles.

      :param el: Pointer to the vector of vector of doubles.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<long>>* el)

      Sets the element value for a vector of vector of longs.

      :param el: Pointer to the vector of vector of longs.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<int>>* el)

      Sets the element value for a vector of vector of ints.

      :param el: Pointer to the vector of vector of ints.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<std::vector<bool>>* el)

      Sets the element value for a vector of vector of bools.

      :param el: Pointer to the vector of vector of bools.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<float>* el)

      Sets the element value for a vector of floats.

      :param el: Pointer to the vector of floats.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<double>* el)

      Sets the element value for a vector of doubles.

      :param el: Pointer to the vector of doubles.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<int>* el)

      Sets the element value for a vector of ints.

      :param el: Pointer to the vector of ints.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<bool>* el)

      Sets the element value for a vector of bools.

      :param el: Pointer to the vector of bools.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<long>* el)

      Sets the element value for a vector of longs.

      :param el: Pointer to the vector of longs.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(std::vector<char>* el)

      Sets the element value for a vector of chars.

      :param el: Pointer to the vector of chars.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(bool* el)

      Sets the element value for a bool.

      :param el: Pointer to the bool.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(double* el)

      Sets the element value for a double.

      :param el: Pointer to the double.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(float* el)

      Sets the element value for a float.

      :param el: Pointer to the float.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(int* el)

      Sets the element value for an int.

      :param el: Pointer to the int.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(long* el)

      Sets the element value for a long.

      :param el: Pointer to the long.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(unsigned long long* el)

      Sets the element value for an unsigned long long.

      :param el: Pointer to the unsigned long long.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(unsigned int* el)

      Sets the element value for an unsigned int.

      :param el: Pointer to the unsigned int.
      :return: True if the element is set, false otherwise.

   .. cpp:member:: bool element(char* el)

      Sets the element value for a char.

      :param el: Pointer to the char.
      :return: True if the element is set, false otherwise.

.. cpp:function:: int count(const std::string* str, const std::string sub)

   Counts the number of occurrences of a substring in a string.

   This function counts how many times a given substring appears within
   a string.

   :param str: A pointer to the string to search.
   :param sub: The substring to count.
   :return: The number of occurrences of the substring in the string.

.. cpp:struct:: element_t

   .. cpp:member:: void set_meta()

      Sets the meta data for the element.

   .. cpp:member:: bool next()

      Moves to the next element.

      :return: True if the move was successful, false otherwise.

   .. cpp:member:: bool boundary()

      Checks if the element is at the boundary.

      :return: True if the element is at the boundary, false otherwise.

.. cpp:struct:: data_t

   .. cpp:member:: data_t()

      Default constructor for the ``data_t`` struct.

   .. cpp:member:: ~data_t()

      Destructor for the ``data_t`` struct.

   .. cpp:member:: void flush()

      Flushes the data.

   .. cpp:member:: void initialize()

      Initializes the data.

   .. cpp:member:: bool next()

      Moves to the next data element.

      :return: True if the move was successful, false otherwise.

   .. cpp:member:: void fetch_buffer()

      Fetches the data buffer.

   .. cpp:member:: void string_type()

      Translates the string type.

.. cpp:struct:: graph_hdf5_w

   .. cpp:member:: void flush_data()

      Flushes the data.

.. cpp:struct:: model_report

   .. cpp:member:: std::string print()

      Prints the model report.

      :return: The string representation of the model report.

   .. cpp:member:: std::string prx(std::map<mode_enum, std::map<std::string, float>>* data, std::string title)

      Helper function to print the model report.

      :param data: Pointer to the data map.
      :param title: The title of the data.
      :return: The string representation of the data.

.. cpp:struct:: optimizer_params_t

   .. cpp:member:: void operator()()

      Overloads the operator() for the ``optimizer_params_t`` struct.

   .. cpp:member:: void set_lr(double*, optimizer_params_t* obj)

      Sets the learning rate.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_lr_decay(double*, optimizer_params_t* obj)

      Sets the learning rate decay.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_weight_decay(double*, optimizer_params_t* obj)

      Sets the weight decay.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_initial_accumulator_value(double*, optimizer_params_t* obj)

      Sets the initial accumulator value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_eps(double*, optimizer_params_t* obj)

      Sets the epsilon value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_betas(std::tuple<float, float>*, optimizer_params_t* obj)

      Sets the betas value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_amsgrad(bool*, optimizer_params_t* obj)

      Sets the amsgrad value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_max_iter(int*, optimizer_params_t* obj)

      Sets the max iter value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_max_eval(int*, optimizer_params_t* obj)

      Sets the max eval value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_tolerance_grad(double*, optimizer_params_t* obj)

      Sets the tolerance grad value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_tolerance_change(double*, optimizer_params_t* obj)

      Sets the tolerance change value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_history_size(int*, optimizer_params_t* obj)

      Sets the history size value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_centered(bool*, optimizer_params_t* obj)

      Sets the centered value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_nesterov(bool*, optimizer_params_t* obj)

      Sets the nesterov value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_alpha(double*, optimizer_params_t* obj)

      Sets the alpha value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_momentum(double*, optimizer_params_t* obj)

      Sets the momentum value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_dampening(double*, optimizer_params_t* obj)

      Sets the dampening value.

      :param obj: Pointer to the ``optimizer_params_t`` object.

   .. cpp:member:: void set_beta_hack(std::vector<float>* val, optimizer_params_t* obj)

      Sets the beta hack value.

      :param val: Pointer to the vector of floats.
      :param obj: Pointer to the ``optimizer_params_t`` object.

.. cpp:struct:: write_t

   .. cpp:member:: void write()

      Writes the data.

   .. cpp:member:: void create(std::string tr_name, std::string path)

      Creates the data.

      :param tr_name: The name of the tree.
      :param path: The path to the file.

   .. cpp:member:: void close()

      Closes the data.

   .. cpp:member:: variable_t* process(std::string* name)

      Processes the data.

      :param name: Pointer to the name of the data.
      :return: Pointer to the ``variable_t`` object.

.. cpp:struct:: writer

   .. cpp:member:: writer()

      Default constructor for the ``writer`` struct.

   .. cpp:member:: ~writer()

      Destructor for the ``writer`` struct.

   .. cpp:member:: void create(std::string* pth)

      Creates the data.

      :param pth: Pointer to the path of the data.

   .. cpp:member:: void write(std::string* tree)

      Writes the data.

      :param tree: Pointer to the tree of the data.

   .. cpp:member:: variable_t* process(std::string* tree, std::string* name)

      Processes the data.

      :param tree: Pointer to the tree of the data.
      :param name: Pointer to the name of the data.
      :return: Pointer to the ``variable_t`` object.

.. comment:: Redundant definitions below are omitted as they duplicate the ones above.

