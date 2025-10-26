.. _selection-template:

SelectionTemplate Methods
-------------------------

This part of the documentation highlights some useful features that are part of the template class.
The selection_template is useful for performing some adhoc studies, such as truth matching or kinematic studies.

.. cpp:class:: selection_template: public tools

   .. cpp:function:: selection_template()

      Selection constructor, which can be overriden by children classes.

   .. cpp:function:: virtual ~selection_template()

      Selection destructor, which can be overriden by children classes.

   .. cpp:function:: void static set_name(std::string*, selection_template*)

      Internal function used to assign the `selection_template` name to the `event_t` struct.

   .. cpp:function:: void static get_name(std::string*, selection_template*)

      Internal function used to retrieve the `selection_template` name from the `event_t` struct.

   .. cpp:function:: void static set_hash(std::string*, selection_template*)

      Internal function used to assign the `selection_template` hash to the `event_t` struct.

   .. cpp:function:: void static get_hash(std::string*, selection_template*)

      Internal function used to retrieve the `selection_template` hash from the `event_t` struct.

   .. cpp:function:: void static get_tree(std::string*, selection_template*)

   .. cpp:function:: void static set_weight(double*, selection_template*)

   .. cpp:function:: void static set_index(long*, selection_template*)

   .. cpp:function:: virtual selection_template* clone()

      Used to clone the current object type. 

   .. cpp:function:: virtual bool selection(event_template* ev)
    
      A function used to select events based on some criteria. 
      If this function returns false, then **strategy** is **NOT** called.


   .. cpp:function:: virtual bool strategy(event_template* ev)

      A function used to define the selection or any analysis methods.

   .. cpp:function:: virtual void merge(selection_template* sel)

      A function reserved to merge any data containers within the selection template.
      For instance, if the class has a vector of floats that need to be merged for each event. 
      An example of how this could be done in code is shown below:

      .. code-block:: C++

         void merge(selection_template* sel){
            // downcast the input selection to the current selection class type.
            selection-name* sx = (selection-name*)sel;
            
            // merge the selection data container, e.g. std::vector<float> + std::vector<float>. 
            // More data types are supported as well.
            merge_data(&this -> some_variable_vector, &sx -> some_variable_vector);     
         }

   .. cpp:function:: void CompileEvent()

      A function which compiles the `selection_template`.
      During compilation two functions are conditionally triggered, namely selection and strategy.
      First the selection function is called, which subsequently triggers the strategy function, if the return value is true.
      Otherwise, the strategy function is not called and the compilation prematurely exits the template.

   .. cpp:function:: selection_template* build(event_template* ev)

   .. cpp:function:: bool operator == (selection_template& p)

      An overloaded equality operator for the selection.

   .. cpp:function:: template<typename g> void sum(std::vector<g*>* ch, particle_template** out)

      A template function used to sum a vector of particle objects without double counting the input vector.
      The output `particle_template` should be a null pointer.

   .. cpp:function:: template<typename g> float sum(std::vector<g*>* ch)

      A template function used to sum a vector of particle objects without double counting.
      The output float is the **Invariant Mass** of the aggregated particle in **GeV**.

   .. cpp:function:: template<typename g> std::vector<g*> vectorize(std::map<std::string, g*>* in)

      A template function used to convert a hash map into a vector of particle objects.
      This function does not check for double counting.

   .. cpp:function::  template<typename g> std::vector<g*> make_unique(std::vector<g*>* inpt)

      A template function used to remove duplicate particle entries in vector.

   .. cpp:function:: template<typename g> void downcast(std::vector<g*>* inpt, std::vector<particle_template*>* out)

      A template function used to convert an arbitrary particle object to a particle_template type.

   .. cpp:function:: template<typename g> void get_leptonics(std::map<std::string, g*> inpt, std::vector<particle_template*>* out)

      A template function used to find leptonic particles.
      For this function to work appropriately, the particle needs to be assigned a PDGID code consistent with either a 
      neutrino or charged lepton.

   .. cpp:function:: template<typename g, typename j> bool contains(std::vector<g*>* inpt, j* pcheck)
    
      A template function used to check whether a given particle object is within the input vector.
      The check uses the hash of the particle.

   .. cpp:var:: cproperty<std::string, selection_template> name

      A property used to assing the `selection_template` a name. 
      Assignment triggers the `set_name` function.

   .. cpp:var:: cproperty<std::string, selection_template> hash

      A property used to fetch the `selection_template` event hash.
      Fetching triggers the `get_hash` function.

   .. cpp:var:: cproperty<std::string, selection_template> tree

   .. cpp:var:: cproperty<long, selection_template> index

   .. cpp:var:: cproperty<double, selection_template> weight

   .. cpp:var:: std::string filename

   .. cpp:var:: event_t data

