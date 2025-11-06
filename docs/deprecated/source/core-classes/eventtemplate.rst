.. _event-template:

EventTemplate Methods
---------------------

This part of the documentation highlights some useful features that are part of the template class. 
The event_template class inherits from the tools class and uses the event_t struct defined under the structs module.

.. cpp:class:: event_template: public tools

   .. cpp:var:: cproperty<std::vector<std::string>, event_template*> trees

      Trees that the framework should scan for the given leaves and branches.

   .. cpp:var:: cproperty<std::vector<std::string>, event_template*> branches

      Branches that the framework should fetch.

   .. cpp:var:: cproperty<std::vector<std::string>, event_template*> leaves

      Leaves that the framework should fetch.

   .. cpp:var:: cproperty<std::string, event_template*> name

      The event template name. This can be arbitrary and optional.

   .. cpp:var:: std::map<std::string, std::string> m_trees

      A public attribute modified by assigning values to `trees`.

   .. cpp:var:: std::map<std::string, std::string> m_branches

      A public attribute modified by assigning values to `branches`.

   .. cpp:var:: std::map<std::string, std::string> m_leaves

      A public attribute modified by assigning values to `leaves`.

   .. cpp:var:: cproperty<std::string, event_template*> hash

      The event hash.
      This value is non-cryptographically generated, but uses the absolute file path + event index.

   .. cpp:var:: cproperty<double, event_template*> weight

      Event weight assigned to the event. By default it is given a value of 1.

   .. cpp:var:: cproperty<long, event_template*> index

      Event index in the ROOT n-tuple.

   .. cpp:var:: event_t data

      A struct used to hold meta data about the event.

   .. cpp:var:: std::string filename

      A string value indicating the original filename from which the event was generated from.

   .. cpp:function:: void static set_trees(std::vector<std::string>*, event_template*)

      A trigger function used by `m_trees` and `trees`. 

   .. cpp:function:: void static set_branches(std::vector<std::string>*, event_template*)

      A trigger function used by `m_branches` and `branches`. 

   .. cpp:function:: void static get_leaves(std::vector<std::string>*, event_template*)

      A trigger function used by `m_leaves` and `leaves`. 

   .. cpp:function:: void add_leaf(std::string key, std::string leaf)

      A trigger function used by `m_leaves` and `leaves`. 

   .. cpp:function:: void static set_name(std::string*, event_template*)

      Sets the name of the event_template implementation. 
      This can be useful when implementing selection_templates.

   .. cpp:function:: void static set_index(long*, event_template*)

      A setter function used to assign the event some index.
      By default the index is the index of the event of a ROOT file.
      This function modifies the `event_t` struct of the `data` attribute.

   .. cpp:function:: void static set_hash(std::string*, event_template*)

      Sets the hash of the event. 
      This value cannot be modified once set and should not be used by default.

   .. cpp:function:: void static get_hash(std::string*, event_template*)

      A getter function used to fetch the current event hash.
      If the hash has not been set, it will generate the hash during the call of `event -> hash`.

   .. cpp:function:: void static set_weight(double*, event_template*)

      A setter function used to assign an event weight to the given event.
      This function is called by assigning a value to `event -> weight` and modifies the `event_t` struct of the `data` attribute.

   .. cpp:function:: virtual event_template* clone()

      A function used to clone the current event implementation. 
      It is important to override this function, so that the current event implementation can be cloned multiple times during event building.
      To use it, simply declare the function as override in the header and create a function with the content `return new <event implementation>`.

   .. cpp:function:: virtual void build(element_t* el)

      A function used to override the current logic of how an event and its particles are built during compilation.
      By default this function should not be overriden or modified unless more customization is required during compilation.

   .. cpp:function:: virtual void CompileEvent()

      A function which can be freely overriden to define how particles should be linked within the event.

   .. cpp:function:: std::map<std::string, event_template*> build_event(std::map<std::string, data_t>* evnt) 

   .. cpp:function:: template <typename G> \
                     void register_particle(std::map<std::string, G*>* particles)

       A template function used to register particles that should be built during compilation.
       **It is important to register any private member variables holding particles as it also cleans up particle pointers following compilation.**

   .. cpp:function:: template <typename G> \
                     void deregister_particle(std::map<std::string, G*>* particles)

   .. cpp:function:: bool operator == (event_template& p)

   .. cpp:function:: void flush_particles()



