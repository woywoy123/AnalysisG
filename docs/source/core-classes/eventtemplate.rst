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

   .. cpp:function:: void static set_index(long*, event_template*)

   .. cpp:function:: void static set_hash(std::string*, event_template*)

   .. cpp:function:: void static get_hash(std::string*, event_template*)

   .. cpp:function:: void static set_weight(double*, event_template*)

   .. cpp:function:: virtual event_template* clone()

   .. cpp:function:: virtual void build(element_t* el)

   .. cpp:function:: virtual void CompileEvent()

   .. cpp:function:: std::map<std::string, event_template*> build_event(std::map<std::string, data_t>* evnt) 

   .. cpp:function:: template <typename G> \
                     void register_particle(std::map<std::string, G*>* particles)

   .. cpp:function:: template <typename G> \
                     void deregister_particle(std::map<std::string, G*>* particles)

   .. cpp:function:: bool operator == (event_template& p)

   .. cpp:function:: void flush_particles()



