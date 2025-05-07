.. _sampletracer_h:

sampletracer.h
==============

This file defines the ``sampletracer`` class, which is a tool for tracing and analyzing data samples.

The ``sampletracer`` class inherits from ``tools`` and ``notification`` classes, providing functionalities
for data processing, analysis, and event notification. It manages a collection of containers, each
associated with a specific file, and provides methods for adding, retrieving, and processing
meta-data, events, graphs, and selections. The class also includes methods for compiling objects,
populating a data loader, and filling selections.

.. cpp:class:: sampletracer : public tools, public notification

    A class for tracing and analyzing data samples.

    The ``sampletracer`` class provides functionalities for managing and processing data samples,
    including meta-data, events, graphs, and selections. It inherits from ``tools`` and
    ``notification`` classes, providing additional functionalities for data processing, analysis,
    and event notification.

    .. cpp:function:: sampletracer()

        Default constructor for the ``sampletracer`` class.

        Initializes a new ``sampletracer`` object with default values.

    .. cpp:function:: ~sampletracer()

        Destructor for the ``sampletracer`` class.

        Releases any resources held by the ``sampletracer`` object.

    .. cpp:function:: bool add_meta_data(meta* meta_, std::string filename)

        Adds meta-data to the container associated with the given filename.

        :param meta* meta_: A pointer to the meta-data object to add.
        :param std::string filename: The filename associated with the container to which the meta-data should be added.
        :return: ``true`` if the meta-data was successfully added, ``false`` otherwise (e.g., if the container already exists).

    .. cpp:function:: meta* get_meta_data(std::string filename)

        Retrieves the meta-data associated with the given filename.

        :param std::string filename: The filename associated with the container from which to retrieve the meta-data.
        :return: A pointer to the meta-data object, or ``nullptr`` if no meta-data is found for the given filename.

    .. cpp:function:: std::vector<event_template*> get_events(std::string label)

        Retrieves a vector of event templates associated with the given label.

        :param std::string label: The label used to identify the events to retrieve.
        :return: A vector of pointers to the event templates associated with the given label.

    .. cpp:function:: void fill_selections(std::map<std::string, selection_template*>* inpt)

        Fills the given map with selection templates from all containers.

        :param std::map<std::string, selection_template*>* inpt: A pointer to the map to fill with selection templates. The keys of the map are strings, and the values are pointers to ``selection_template`` objects.

    .. cpp:function:: bool add_event(event_template* ev, std::string label)

        Adds an event template to the container associated with the event's filename.

        :param event_template* ev: A pointer to the event template to add.
        :param std::string label: The label to associate with the event template.
        :return: ``true`` if the event template was successfully added, ``false`` otherwise.

    .. cpp:function:: bool add_graph(graph_template* gr, std::string label)

        Adds a graph template to the container associated with the graph's filename.

        :param graph_template* gr: A pointer to the graph template to add.
        :param std::string label: The label to associate with the graph template.
        :return: ``true`` if the graph template was successfully added, ``false`` otherwise.

    .. cpp:function:: bool add_selection(selection_template* sel)

        Adds a selection template to the container associated with the selection's filename.

        :param selection_template* sel: A pointer to the selection template to add.
        :return: ``true`` if the selection template was successfully added, ``false`` otherwise.

    .. cpp:function:: void populate_dataloader(dataloader* dl)

        Populates the given data loader with data from all containers.

        :param dataloader* dl: A pointer to the data loader to populate.

    .. cpp:function:: void compile_objects(int threads)

        Compiles the objects in all containers using the specified number of threads.

        This method iterates through the root containers, sets the output path for each container,
        and then compiles the objects within each container using a thread pool. It utilizes
        lambdas for the compilation task and flushing titles. A progress bar is displayed
        during the compilation process if the ``shush`` flag is not set.

        :param int threads: The number of threads to use for compilation.

    .. cpp:member:: std::string* output_path = nullptr

        A pointer to the output path for the compiled objects.

    .. cpp:member:: std::map<std::string, container> root_container

        A map of filenames to containers, storing the data associated with each file. (Private Member)

