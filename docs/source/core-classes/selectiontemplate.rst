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

   .. cpp:function:: void static set_hash(std::string*, selection_template*)

   .. cpp:function:: void static get_hash(std::string*, selection_template*)

   .. cpp:function:: void static get_tree(std::string*, selection_template*)

   .. cpp:function:: void static set_weight(double*, selection_template*)

   .. cpp:function:: void static set_index(long*, selection_template*)

   .. cpp:function:: virtual selection_template* clone()

   .. cpp:function:: virtual bool selection(event_template* ev)

   .. cpp:function:: virtual bool strategy(event_template* ev)

   .. cpp:function:: virtual void merge(selection_template* sel)

   .. cpp:function:: void CompileEvent()

   .. cpp:function:: selection_template* build(event_template* ev)

   .. cpp:function:: bool operator == (selection_template& p)

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






