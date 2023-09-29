.. _sample-tracer:

The SampleTracer
****************

A core module which is being inherited by all generator classes. 
This class focuses on tracking sample generation including events, graphs and selections, and how they are cached in memory. 
The class can be used as a standalone package, but is primarily intended to be integrated into any abstract generator class that one might want to implement, this will be illustrated later.
Most of the functionalities, such as the magic functions are implemented within this class, making it a rather useful core module. 

Methods and Attributes
______________________

.. py:class:: SampleTracer

    .. py:method:: __getstate__ -> tracer_t

        Export this tracer including all samples (selections, graphs, events) and state (settings).

    .. py:method:: __setstate__(tracer_t inpt)

        Import tracer parameters including all samples (selections, graphs, events) and state (settings).

    .. py:method:: __getitem__(key: Union[list, str]) -> bool or list

        Scan indexed content and return a list of matches or a boolean if nothing has been found.

    .. py:method:: __contains__(str val) -> bool

        Check if query is in sample tracer.

    .. py:method:: __len__ -> int

        Return length of the entire sample.

    .. py:method:: __add__(SampleTracer other) -> SampleTracer

        Add two SampleTracers to create an independent SampleTracer. 
        Content of both samples is compared and summed as a set. 

    .. py:method:: __radd__(other) -> SampleTracer

    .. py:method:: __iadd__(SampleTracer other) -> SampleTracer

        Add the incoming tracer to this tracer, i.e. append content.

    .. py:method:: __iter__

        Iteratate over the Sample Tracer with given parameters, e.g. cache type etc.

    .. py:method:: __next__ -> Event

        The return of the iterator is an Event (Not to be confused with EventTemplate). 
        This **Event** is a batched version of **SelectionTemplate**/**GraphTemplate**/**EventTemplate** and **MetaData**


    .. py:method:: preiteration -> bool
        
        A place holder for adding last minute behaviour changes to the iteration process.
        This can include loading specific caches or changing general behaviour, i.e. pre-fetching etc.

    .. py:method:: DumpTracer(retag: Union[str, None]) -> None

        Preserve the index map of the samples within the tracer.
        The output of this is a set of HDF5 files, which are written in the form of their Logical File Names or original sample name.

        :param str, None retag: Allows for tagging specific samples of the tracer to be tagged.


    .. py:method:: RestoreTracer(dict tracers = {}, sample_name: Union[None, str]) -> None

         Restore the index map of the samples within the tracer.

         :param dict tracers: Restore these HDF5 file directories
         :param None, str sample_name: Restore only tracer samples with a particular sample name tag.

    .. py:method:: DumpEvents -> None
        
        Preserve the **EventTemplates** in HDF5 files.

    .. py:method:: DumpGraphs -> None

        Preserve the **GraphTemplates** in HDF5 files.

    .. py:method:: DumpSelections -> None

        Preserve the **SelectionTemplates** in HDF5 files.

    .. py:method:: RestoreEvents(list these_hashes = []) -> None

        Restore **EventTemplates** matching a particular set of hashes.
        :params list these_hashes: A list of hashes consistent with events indexed by the tracer.

    .. py:method:: RestoreGraphs(list these_hashes = []) -> None

        Restore **GraphTemplates** matching a particular set of hashes.
        :params list these_hashes: A list of hashes consistent with events indexed by the tracer.

    .. py:method:: RestoreSelections(list these_hashes = []) -> None

        Restore **SelectionTemplates** matching a particular set of hashes.
        :params list these_hashes: A list of hashes consistent with events indexed by the tracer.

    .. py:method:: FlushEvents(list these_hashes = []) -> None

        Delete **EventTemplates** matching a particular set of hashes from RAM
        :params list these_hashes: A list of hashes consistent with events indexed by the tracer.

    .. py:method:: FlushGraphs(list these_hashes = []) -> None

        Delete **GraphsTemplates** matching a particular set of hashes from RAM.
        :params list these_hashes: A list of hashes consistent with events indexed by the tracer.

    .. py:method:: FlushSelections(list these_hashes = []) -> None

        Delete **SelectionTemplates** matching a particular set of hashes from RAM.
        :params list these_hashes: A list of hashes consistent with events indexed by the tracer.

    .. py:method:: _makebar(inpt: Union[int], CustTitle: Union[None, str] = None) -> (None, tqdm)

        Creates a *tqdm* progress bar.
        :params int inpt: Length of the sample, i.e. the range of the bar.
        :params None, str CustTitle: Override the default progress prefix title (see **Caller**).

    .. py:method:: trace_code(obj) -> code_t

        Preserve an object which is independent of the current file implementation (see **Code**).
        :params obj: Any Python object

    .. py:method:: rebuild_code(val: Union[list, str, None]) -> list[Code]

        Rebuild a set of **Code** objects which mimic the originally traced code.
        :params list, str, None val: Rebuild these strings from the traced code of the SampleTracer.

    .. py:method:: ImportSettings(settings_t inpt) -> None

        Apply settings from the input to the current SampleTracer.
        :params settings_t inpt: A dictionary like object with specific keys. See the **Data Type and Dictionary Section**.

    .. py:method:: ExportSettings -> settings_t

        Export the current settings of the SampleTracer.

    .. py:method:: clone -> SampleTracer

        Returns a copy of the current object SampleTracer object.
        This will **NOT** clone the content of the source tracer.

    .. py:method:: is_self(inpt, obj = SampleTracer) -> bool

        Checks whether the input has a type consistent with the object type (also inherited objects are permitted).
        :params inpt: Any Python object
        :params obj: The target object type to check against, e.g. SampleTracer type.

    .. py:method:: makehashes() -> dict
        
        Returns a dictionary of current hashes not found in RAM.

    .. py:method:: makelist() -> list[Event]

        Returns a list of **Event** objects regardless if Templates are not loaded in memory.


    .. py:method:: AddEvent(event_inpt, meta_inpt = None) -> None

    .. py:method:: AddGraph(graph_inpt, meta_inpt = None) -> None

    .. py:method:: AddSelections(selection_inpt, meta_inpt = None) -> None

    .. py:method:: SetAttribute(fx, str name) -> bool


    .. py:attribute:: Tree -> str

        Returns current ROOT Tree being used.

    .. py:attribute:: ShowTrees -> list[str]

        Returns a list of ROOT Trees found within the index.

    .. py:attribute:: Event -> EventTemplate or Code

        Specifies the an **EventTemplate** inherited event implementation to use for building Event objects from ROOT Files.
 
    .. py:attribute:: ShowEvents -> list[str]

         Returns a list of **EventTemplate** implementations found within the index.

    .. py:attribute:: GetEvent -> bool

        Forcefully get or ignore **EventTemplate** types from the **Event** object.
        This is useful to avoid redundant sample fetching from RAM.

    .. py:attribute:: EventCache -> bool

        Specifies whether to generate a cache after constructing **Event** objects. 
        If this is enabled without specifying a **ProjectName**, a folder called **UNTITLED** is generated.


    .. py:attribute:: EventName -> str

    .. py:attribute:: Graph -> GraphTemplate or Code

        Specifies the event graph implementation to use for constructing graphs.

    
    .. py:attribute:: ShowGraphs -> list[str]

    .. py:attribute:: GetGraph -> bool

    .. py:attribute:: DataCache -> bool

        Specifies whether to generate a cache after constructing graph objects. 
        If this is enabled without having an event cache, the **Event** attribute needs to be set. 

    .. py:attribute:: GraphName -> str

    .. py:attribute:: Selections -> dict[str, SelectionTemplate or Code]

    .. py:attribute:: ShowSelections -> list[str]

    .. py:attribute:: GetSelection -> bool

    .. py:attribute:: SelectionName -> str

    .. py:attribute:: Optimizer -> str

        Expects a string of the specific optimizer to use.
        Current choices are; **SGD** - Stochastic Gradient Descent and **ADAM**.

    .. py:attribute:: Scheduler -> str

        Expects a string of the specific scheduler to use. 
        Current choices are
        - **ExponentialLR** 
        - **CyclicLR**

    .. py:attribute:: Model -> ModelWrapper or Code

        The target model to be trained. 

    .. py:attribute:: OptimizerParams -> dict

        A dictionary containing the specific input parameters for the chosen **Optimizer**.

    .. py:attribute:: SchedulerParams -> dict

        A dictionary containing the specific input parameters for the chosen **Scheduler**.

    .. py:attribute:: ModelParams -> dict

    .. py:attribute:: kFold -> list[int]

        Explicitly use these kFolds during training. 
        This can be quite useful when doing parallel traning, since each kFold is trained completely independently. 
        The variable can be set to a single integer or list of integers


    .. py:attribute:: Epoch -> int

    .. py:attribute:: kFolds -> int

        Number of folds to use for training


    .. py:attribute:: Epochs -> int

    .. py:attribute:: BatchSize -> int
    
        How many Graphs to group into a single big graph (also known as batch training).

    .. py:attribute:: GetAll -> bool

    .. py:attribute:: nHashes -> int

    .. py:attribute:: ShowLength -> dict

    .. py:attribute:: EventStart -> int or None

        The event to start from given a set of ROOT samples. 
        Useful for debugging specific events.



    .. py:attribute:: EventStop -> int or None

        The number of events to generate. 

    .. py:attribute:: EnablePyAMI -> bool

    .. py:attribute:: Files -> dict

    .. py:attribute:: SampleMap -> dict


    .. py:attribute:: ProjectName -> str

        Specifies the output folder of the analysis. 
        If the folder is non-existent, a folder will be created.

    .. py:attribute:: OutputDirectory -> str

        Specifies the output directory of the analysis. 
        This is useful if the output needs to be placed outside of the working directory.


    .. py:attribute:: WorkingPath -> str

        Returns the current working path of the Analysis.
        Constructed as; **OutputDirectory/ProjectName**

    .. py:attribute:: RunName -> str

        The name given to the particular training session of the Graph Neural Network.

    .. py:attribute:: Caller -> str

        A string controlling the verbose information prefix.

    .. py:attribute:: Verbose -> int

        An integer which increases the verbosity of the framework, with 3 being the highest and 0 the lowest.

    .. py:attribute:: DebugMode -> bool

        Expects a boolean, if this is set to **True**, a complete print out of the training is displayed. 

    .. py:attribute:: Chunks -> int

        An integer which regulates the number of entries to process for each given core. 
        This is particularly relevant when constructing events, as to avoid memory issues. 
        As an example, if Threads is set to 2 and **chnk** is set to 10, then 10 events will be processed per core. 
    .. py:attribute:: Threads -> int

        The number of CPU threads to use for running the framework.
        If the number of threads is set to 1, then the framework will not print a progress bar.

    .. py:attribute:: Device -> str

        The device used to run ``PyTorch`` training on.
        Options are ``cuda`` or ``cpu``.

    .. py:attribute:: TrainingName -> str

        Name of the training sample to be used. 

    .. py:attribute:: SortByNodes -> bool

        Sort the input graph sample by nodes.
        This is useful when the model is node agnostic, but requires recomputation of internal variables based on variable graph node sizes.
        For instance, when computing the combinatorial of a graph, it is faster to compute the combinations for n-nodes and batch n-sized graphs into a single sample set.

    .. py:attribute:: ContinueTraining -> bool

        Whether to continue the training from the last known checkpoint (after each epoch).

    .. py:attribute:: KinematicMap -> dict

        place holder

    .. py:attribute:: EnableReconstruction -> bool

        place holder

    .. py:attribute:: PlotLearningMetrics -> bool

        Whether to output various metric plots whilst training.
        This can be enabled before training or re-run after training from the training cache.
