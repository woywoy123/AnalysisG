.. cpp:file:: container.dox
    :brief: Implementation file for the container class and entry_t struct.

    This file provides the definitions for the methods declared in container.h.
    It includes the constructor, destructor, and various methods for managing
    analysis objects like events, graphs, and selections within the container.
    It also implements the entry_t struct methods for managing collections
    of these objects associated with a specific hash.

.. cpp:struct:: entry_t
    :brief: Represents a collection of analysis objects associated with a unique hash.

    This struct holds vectors of pointers to different template types (event, graph, selection)
    that share the same origin or processing characteristics, identified by a hash string.
    It also stores the results of graph compilation (:cpp:member:`m_data`).

    .. cpp:member:: std::string hash = ""
        :brief: Unique identifier string for this entry.

    .. cpp:member:: std::vector<graph_t*> m_data = {}
        :brief: Vector storing pointers to compiled graph data (`graph_t`). Ownership might be transferred.

    .. cpp:member:: std::vector<graph_template*> m_graph = {}
        :brief: Vector storing pointers to graph templates associated with this hash.

    .. cpp:member:: std::vector<event_template*> m_event = {}
        :brief: Vector storing pointers to event templates associated with this hash.

    .. cpp:member:: std::vector<selection_template*> m_selection = {}
        :brief: Vector storing pointers to selection templates associated with this hash.

    .. cpp:function:: void init()
        :brief: Initializes the internal vectors.
        Reserves space in the vectors to potentially avoid reallocations for small numbers of items.

    .. cpp:function:: void destroy()
        :brief: Destroys the template objects pointed to by the vectors.
        Calls the templated destroy method for each vector containing pointers
        (:cpp:member:`m_event`, :cpp:member:`m_graph`, :cpp:member:`m_selection`) to delete the pointed-to objects and clear the vectors.
        Note: Does not clear :cpp:member:`m_data` as ownership might be transferred (e.g., to dataloader).

    .. cpp:function:: bool has_event(event_template* ev)
        :brief: Checks for existence and potentially adds an event template.
        Iterates through the :cpp:member:`m_event` vector. If a match is found based on the ``tree``
        and ``name`` members, it returns true. Otherwise, it adds the provided event
        pointer ``ev`` to the :cpp:member:`m_event` vector and returns false.

        :param ev: Pointer to the event_template to check and potentially add.
        :return: True if an event with the same tree and name was already present, false otherwise.

    .. cpp:function:: bool has_graph(graph_template* gr)
        :brief: Checks for existence and potentially adds a graph template.
        Iterates through the :cpp:member:`m_graph` vector. If a match is found based on the ``tree``
        and ``name`` members, it returns true. Otherwise, it adds the provided graph
        pointer ``gr`` to the :cpp:member:`m_graph` vector and returns false.

        :param gr: Pointer to the graph_template to check and potentially add.
        :return: True if a graph with the same tree and name was already present, false otherwise.

    .. cpp:function:: bool has_selection(selection_template* sel)
        :brief: Checks for existence and potentially adds a selection template.
        Iterates through the :cpp:member:`m_selection` vector. If a match is found based on the ``tree``
        and ``name`` members, it returns true. Otherwise, it adds the provided selection
        pointer ``sel`` to the :cpp:member:`m_selection` vector and returns false.

        :param sel: Pointer to the selection_template to check and potentially add.
        :return: True if a selection with the same tree and name was already present, false otherwise.

    .. cpp:function:: template <typename g> void destroy(std::vector<g*>* c)
        :brief: Template function to delete objects pointed to by a vector and clear the vector.
        Iterates through the vector, deletes each element (assuming it's a pointer),
        sets the pointer to nullptr, and then swaps the vector with an empty one to release memory.

        :tparam g: The type of the pointer stored in the vector.
        :param c: Pointer to the std::vector<g*> to be cleared and whose elements are to be deleted.


.. cpp:class:: container : public tools
    :brief: Manages collections of analysis objects (events, graphs, selections) and orchestrates their processing.

    Inherits from ``tools`` (presumably providing utility functions like ``split``, ``create_path``).
    Acts as a central hub during the analysis process, grouping related analysis templates
    based on a hash (often derived from event context) using the :cpp:struct:`entry_t` struct.
    It handles adding templates, associating metadata, compiling the templates, merging selection results,
    and populating dataloaders with graph data.

    .. cpp:function:: container()
        :brief: Default constructor.
        Initializes a new container object.

    .. cpp:function:: ~container()
        :brief: Destructor. Cleans up allocated resources.
        Responsible for cleaning up all dynamically allocated memory.
        This includes deleting the metadata object, the filename string,
        destroying all entries in the :cpp:member:`random_access` map, and cleaning up
        the :cpp:member:`merged` selections map if it exists.

    .. cpp:function:: void add_meta_data(meta* data, std::string fname)
        :brief: Adds metadata and associated filename.
        Stores the pointer to the metadata object and creates a copy of the filename string.

        :param data: Pointer to the meta object containing metadata.
        :param fname: The filename associated with the data being processed.

    .. cpp:function:: meta* get_meta_data()
        :brief: Retrieves the stored metadata pointer.

        :return: :cpp:class:`meta`\* Pointer to the stored meta object. Returns nullptr if no metadata has been added.

    .. cpp:function:: bool add_selection_template(selection_template* sel)
        :brief: Adds a selection template to the appropriate entry.
        It retrieves or creates the entry corresponding to the selection's hash using :cpp:func:`add_entry`.
        Associates the container's metadata with the selection.
        Checks if a selection with the same name and tree already exists within the entry using :cpp:func:`entry_t::has_selection`.
        If not, adds the selection to the entry's selection list.

        :param sel: Pointer to the selection_template object to add.
        :return: True if an identical selection (same name, same tree) was already present in the entry, false otherwise.

    .. cpp:function:: bool add_event_template(event_template* ev, std::string _label)
        :brief: Adds an event template to the appropriate entry.
        If the container doesn't have a label yet (:cpp:member:`label`), it adopts the provided label.
        It retrieves or creates the entry corresponding to the event's hash using :cpp:func:`add_entry`.
        Associates the container's metadata with the event.
        Checks if an event with the same name and tree already exists within the entry using :cpp:func:`entry_t::has_event`.
        If not, adds the event to the entry's event list.

        :param ev: Pointer to the event_template object to add.
        :param _label: The label associated with this event source.
        :return: True if an identical event (same name, same tree) was already present in the entry, false otherwise.

    .. cpp:function:: bool add_graph_template(graph_template* gr, std::string _label)
        :brief: Adds a graph template to the appropriate entry.
        If the container doesn't have a label yet (:cpp:member:`label`), it adopts the provided label.
        It retrieves or creates the entry corresponding to the graph's hash using :cpp:func:`add_entry`.
        Associates the container's metadata with the graph.
        Checks if a graph with the same name and tree already exists within the entry using :cpp:func:`entry_t::has_graph`.
        If not, adds the graph to the entry's graph list.

        :param gr: Pointer to the graph_template object to add.
        :param _label: The label associated with this graph source.
        :return: True if an identical graph (same name, same tree) was already present in the entry, false otherwise.

    .. cpp:function:: void fill_selections(std::map<std::string, selection_template*>* inpt)
        :brief: Merges compiled selections into an external map.
        If the container has compiled selections stored in its :cpp:member:`merged` map, this function
        iterates through them. For each selection in :cpp:member:`merged`, it finds the corresponding
        selection in the input map ``inpt`` and merges the results using the ``merger`` method.
        After merging, the selection object in the container's :cpp:member:`merged` map is deleted.
        Finally, the container's :cpp:member:`merged` map itself is cleared and deleted.

        :param inpt: Pointer to the target map (typically a global or aggregator map) where selection results should be merged into.

    .. cpp:function:: void get_events(std::vector<event_template*>* out, std::string label)
        :brief: Retrieves event templates, optionally filtered by label.
        If the provided label matches the container's :cpp:member:`label` (or if the provided label is empty),
        it iterates through all entries in the :cpp:member:`random_access` map and appends their
        event_template pointers to the output vector ``out``.

        :param out: Pointer to a std::vector<event_template*> where the found event pointers will be added.
        :param label: The label to filter events by. If empty or matches the container's label, all events are considered.

    .. cpp:function:: void populate_dataloader(dataloader* dl)
        :brief: Populates a dataloader with compiled graph data.
        Iterates through all entries in the :cpp:member:`random_access` map. For each entry,
        it iterates through the stored ``graph_t`` objects (results of graph compilation)
        in :cpp:member:`entry_t::m_data` and passes each one to the ``dataloader``'s ``extract_data`` method.
        After processing an entry, its :cpp:member:`entry_t::m_data` vector is cleared.
        Finally, the container's :cpp:member:`random_access` map is cleared, releasing the entries.

        :param dl: Pointer to the dataloader object to populate.

    .. cpp:function:: void compile(size_t* len, int threadIdx)
        :brief: Compiles all stored templates (events, graphs, selections).
        This method iterates through all entries in the :cpp:member:`random_access` map.
        For each entry:
        1. Compiles associated event templates (``CompileEvent``).
        2. Handles selection templates:
            - Initializes the :cpp:member:`merged` map if selections exist and it's not already created.
            - For the first entry processed, creates output file handles (``write_t``) for each selection if an :cpp:member:`output_path` is set.
            - Clones the selection templates from the first entry into the :cpp:member:`merged` map, assigning the thread index and file handle.
            - Compiles each selection template (``CompileEvent``), potentially performing bulk writes.
            - Merges the results of the compiled selection into the corresponding template in the :cpp:member:`merged` map.
            - Writes data if output path is set and bulk writing is not enabled (``handles[name]->write()``).
        3. Handles graph templates:
            - Assigns the thread index.
            - Executes preselection if defined.
            - Compiles the graph template (``CompileEvent``).
            - Exports the compiled graph data (``data_export``).
            - Stores the exported graph data (``graph_t``) within the entry's :cpp:member:`entry_t::m_data` vector.
        4. Destroys the processed entry's internal template vectors (:cpp:func:`entry_t::destroy()`).
        5. Increments the processed entry counter ``l``.
        After processing all entries:
        6. Finalizes bulk writing for merged selections.
        7. Closes and deletes all output file handles.
        8. Updates the counter ``l`` to the total number of entries processed.

        :param len: Pointer to a size_t variable, used to track the number of processed entries. It's updated during and at the end of the function.
        :param threadIdx: The index of the current processing thread, passed down to templates.

    .. cpp:function:: size_t len()
        :brief: Returns the number of entries in the container.

        :return: size_t The number of key-value pairs in the :cpp:member:`random_access` map.

    .. cpp:function:: entry_t* add_entry(std::string hash)
        :brief: Adds or retrieves an entry based on hash.
        If an entry with the given hash already exists in :cpp:member:`random_access`, a pointer to it is returned.
        Otherwise, a new :cpp:struct:`entry_t` is created, initialized (:cpp:func:`entry_t::init`), assigned the hash,
        inserted into the map, and a pointer to the new entry is returned.

        :param hash: The unique hash string identifying the entry.
        :return: :cpp:struct:`entry_t`\* Pointer to the existing or newly created entry_t object.

    .. cpp:member:: meta* meta_data = nullptr
        :brief: Pointer to the metadata object associated with this container's data.

    .. cpp:member:: std::string* filename = nullptr
        :brief: Pointer to the string holding the filename of the original data source.

    .. cpp:member:: std::string* output_path = nullptr
        :brief: Pointer to the string specifying the base output path for results (e.g., selections). If nullptr, output might be disabled.

    .. cpp:member:: std::string label = ""
        :brief: Label associated with the data source (e.g., sample type, dataset name). Used for organizing output.

    .. cpp:member:: std::map<std::string, entry_t> random_access
        :brief: Map storing :cpp:struct:`entry_t` objects, keyed by their hash string.
        Provides random access to collections of templates based on their shared hash.

    .. cpp:member:: std::map<std::string, selection_template*>* merged = nullptr
        :brief: Map storing pointers to merged selection templates, keyed by selection name.
        This map holds the aggregated results of selections after the compile step.
        It is initialized during :cpp:func:`compile` if selections are present and potentially written out.
        It is cleared and deleted after its contents are transferred (e.g., via :cpp:func:`fill_selections`).

