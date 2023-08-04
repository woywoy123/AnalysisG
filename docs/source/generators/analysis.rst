This section is still under construction.....

The Analysis Generator
**********************
The Analysis class is a module of the framework and serves as the main interface, which unifies all modules found within this package. 
It is composed of several generator modules and binds for instance the event, graph and selection generation into a single and convenient interface.
This class can also be used to generate and automate **Condor**/**DAGMAN** scripting, further simplifying a given analysis pipeline. 

This class has a large number of parameters, so it is advised to study this part of the documentation carefully, such that it's tools can be used as effectively as possible. 

Introduction
____________


Magic Functions
_______________

.. code-block:: python 

    # Iteration
    [i for i in Analysis]

    # Length operator
    len(Analysis)

    # Summation operator 
    Analysis3 = Analysis1 + Analysis2
    AnalysisSum = [Analysis1, Analysis2, Analysis3]

    # These will be equal since the events and ROOT files 
    # will be merged into a summed Analysis.
    # This means that if Analysis1 and Analysis2 have different events/ROOT Files
    # then the sum of these two will give Analysis3, but summing Analysis1 and 2 
    # again wont result in double counting.
    len(AnalysisSum) == len(Analysis3)


Minimalistic Example
____________________

.. code-block:: python

    from AnalysisG import Analysis
    from SomeEventImplementation import CustomEvent

    Ana = Analysis()
    Ana.ProjectName = "Example"
    Ana.InputSample(<name of sample>, "/some/sample/directory")
    Ana.Event = CustomEvent
    Ana.EventCache = True
    Ana.Launch
 
    for event in Ana:
        print(event)

   
Run-Time Attributes
___________________

- ``Verbose``: 
    An integer which increases the verbosity of the framework, with 3 being the highest and 0 the lowest.

- ``Threads``: 
    The number of CPU threads to use for running the framework.
    If the number of threads is set to 1, then the framework will not print a progress bar. 

- ``chnk``: 
    An integer which regulates the number of entries to process for each given core. 
    This is particularly relevant when constructing events, as to avoid memory issues. 
    As an example, if Threads is set to 2 and **chnk** is set to 10, then 10 events will be processed per core. 

- ``ProjectName``: 
    Specifies the output folder of the analysis. If the folder is non-existent, a folder will be created.

- ``OutputDirectory``: 
    Specifies the output directory of the analysis. This is useful if the output needs to be placed outside of the working directory.

Event Generation Attributes
___________________________

- ``Event``: 
    Specifies the event implementation to use for constructing the Event objects from ROOT Files.

- ``EventStart``: 
    The event to start from given a set of ROOT samples. Useful for debugging specific events.

- ``EventStop``: 
    The number of events to generate. 

- ``EventCache``: 
    Specifies whether to generate a cache after constructing **Event** objects. 
    If this is enabled without specifying a **ProjectName**, a folder called **UNTITLED** is generated.

Graph Generation Attributes
___________________________

- ``EventGraph``:
    Specifies the event graph implementation to use for constructing graphs.

- ``SelfLoop``:
    Given an event graph implementation, add edges to particle nodes which connect to themselves.

- ``FullyConnect``:
    Given an event graph implementation, create a fully connected graph.

- ``DataCache``:
    Specifies whether to generate a cache after constructing graph objects. 
    If this is enabled without having an event cache, the **Event** attribute needs to be set. 

- ``TestFeature``: 
    A parameter mostly concerning graph generation. 
    It checks whether the supplied features are compatible with the **Event** python object. 
    If any of the features fail, an alert is issued. 

Optimizer Attributes
____________________

- ``TrainingPercentage``:
    Assign some percentage to training and reserve the remaining for testing.

- ``kFolds``:
    Number of folds to use for training 

- ``kFold``:
    Explicitly use this kFold during training. 
    This can be quite useful when doing parallel traning, since each kFold is trained completely independently. 

- ``BatchSize``:
    How many Event Graphs to group into a single graph.

- ``Model``:
    The model to be trained, more on this later.

- ``DebugMode``:
    Expects a boolean, if this is set to **True**, a complete print out of the training is displayed. 

- ``ContinueTraining``:
    Whether to continue the training from the last known checkpoint (after each epoch).

- ``Optimizer``:
    Expects a string of the specific optimizer to use.
    Current choices are; **SGD** - Stochastic Gradient Descent and **ADAM**.

- ``OptimizerParams``: 
    A dictionary containing the specific input parameters for the chosen **Optimizer**.

- ``Scheduler``:
    Expects a string of the specific scheduler to use. 
    Current choices are
    - **ExponentialLR** 
    - **CyclicLR**

- ``SchedulerParams``: 
    A dictionary containing the specific input parameters for the chosen **Scheduler**.

- ``Device``: 
    The device used to run ``PyTorch`` training on. This also applies to where to store graphs during compilation.

Run-Time Functions:
___________________
 
- ``InputSample(Name, SampleDirectory)``:
    This function is used to specify the directory or sample to use for the analysis. 
    The **Name** parameter expects a string, which assigns a name to **SampleDirectory** and is used for book-keeping. 
    **SampleDirectory** can be either a string, pointing to a ROOT file or a nested dictionary with keys indicating the path and values being a string or list of ROOT files. 

- ``AddSelection(Name, inpt)``:
    The **Name** parameter specifies the name of the selection criteria, for instance, **MySelection**. 
    The **inpt** specifies the **Selection** implementation to use, more on this later. 

- ``MergeSelection(Name)``:
    This function allows for post selection output to be merged into a single pickle file. 
    During the execution of the **Selection** implementation, multiple threads are spawned, which individually save the output of each event selection, meaning a lot of files being written and making it less ideal for inspecting the data.
    Merging combines all the internal data into one single file and deletes files being merged. 
 
- ``DumpSettings``:
    Returns a directory of the settings used to configure the **Analysis** object. 

- ``ImportSettings(inpt)``:
    Expects a dictionary of parameters used to configure the object.

- ``Quantize(inpt, size)``:
    Expects a dictionary with lists of ROOT files, that need to be split into smaller lists (defined by size).
    For instance, given a size of 2, a list of 100 ROOT files will be split into 50 lists with length 2.

- ``Launch``:
    Launches the Analysis with the specified parameters.
 
