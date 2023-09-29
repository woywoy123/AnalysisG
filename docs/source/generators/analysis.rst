The Analysis Generator
**********************

The Analysis class is a module of the framework and serves as the main interface, which unifies all modules within the framework into one interface.
It is composed of several generator modules and binds for instance the event, graph and selection generation together as a chained pipeline.
This class can also be used to generate and automate **Condor**/**DAGman** scripting, further simplifying a given analysis pipeline. 

This class has a large number of parameters, so it is advised to study this part of the documentation carefully, such that it's tools can be used as effectively as possible. 

Generating Events from EventTemplates:
______________________________________

The first step is to initialize the ``Analysis`` instance and providing it with a sample path/directory. 
This should look something like the code shown below: 

.. code-block:: python 
    
    from AnalysisG import Analysis

    Ana = Analysis()

    # Scan the entire "Some/Path/To/ROOT/" path for .root files
    Ana.InputSample("SomeName", "Some/Path/To/ROOT/")

    # Or point at a single ROOT file in "Some/Path/To/ROOT/"
    Ana.InputSample("SomeName", "Some/Path/To/ROOT/One.root")

    # Or use some dictionary for multiple paths to scan
    scans = {
        "Some/Path/To/ROOT1" : ["*"],
        "Some/Path/To/ROOT2" : ["One.root", "Two.root", "Three.root"]
    }
    Ana.InputSample("SomeName", scans)


The purpose of ``SomeName`` is to assign certain samples a name. 
This could be useful if say, there are mulitple ROOT files which have previously been sourced/compiled, but the aim is to run only a specific sample type. 
For instance, if the framework was used to generate events from :math:`t\bar{t}` and :math:`t\bar{t}t\bar{t}`, and you wanted to only load :math:`t\bar{t}`, then one could specify the sample name and only :math:`t\bar{t}` would be loaded.

The next step would be to point the framework to the event implementation (see :ref:`event-start` for an introduction). 
Continuing on from the above code, this would be done as such: 

.. code-block:: python 

   from PathToCustomEvent.Event import CustomEvent

   Ana.Event = CustomEvent # <--- link the Event
   Ana.EventCache = True

From the above, the command ``EventCache`` informs the framework to save the Python representation ``CustomEvent`` into HDF5 files. 
This significantly reduces computational time, since the framework would not need to always re-read ROOT files and convert them into ``CustomEvent``.
To start the process, simply call ``Ana.Launch()`` and the instance will do some preliminary checks. 
At some point in the run-time, an error/warning informing you about not having a **VOMS** session is issued, this is totally benign and can be safely ignored.

To speed up the compilation step, the number of threads or chunks per thread can be modified.
Unfortunately, these parameters are not automatically optimized, so setting these parameters is completely arbitrary. 
As the name suggests, ``Threads`` refers to the number of ``CPU Threads``, and is set to 6 by default. 
The ``Chunks`` parameter is a variable which forces the framework to allocate a number of events per CPU thread.
For instance, if 1000 events are being processed, one could distribute 100 events across 10 CPU threads, thus increasing parallelism. 
There is one subtle drawback with a high number of jobs per thread, and this is exessive RAM usage and transfer speeds between the parent and children threads. 
This is due to Python's threading model leveraging the ``pickle`` protocol to transfer across child and parent threads.


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
    Ana.Launch()
 
    for event in Ana:
        print(event)

  
For a full set of attributes, consult the **SampleTracer** documentation section. 
Attributes listed below are exclusive settings not associated with the **SampleTracer**.

Graph Generation Attributes
___________________________

- ``TestFeature``: 
    A parameter mostly concerning graph generation. 
    It checks whether the supplied features are compatible with the **Event** python object. 
    If any of the features fail, an alert is issued. 

Optimizer Attributes
____________________

- ``TrainingPercentage``:
    Assign some percentage to training and reserve the remaining for testing.

Run-Time Functions:
___________________
 
- ``InputSample(Name, SampleDirectory)``:
    This function is used to specify the directory or sample to use for the analysis. 
    The **Name** parameter expects a string, which assigns a name to **SampleDirectory** and is used for book-keeping. 
    **SampleDirectory** can be either a string, pointing to a ROOT file or a nested dictionary with keys indicating the path and values being a string or list of ROOT files. 

- ``AddSelection(inpt)``:
    The **inpt** specifies the **Selection** implementation to use, more on this later. 
 
- ``Quantize(inpt, size)``:
    Expects a dictionary with lists of ROOT files, that need to be split into smaller lists (defined by size).
    For instance, given a size of 2, a list of 100 ROOT files will be split into 50 lists with length 2.

- ``Launch``:
    Launches the Analysis with the specified parameters.

