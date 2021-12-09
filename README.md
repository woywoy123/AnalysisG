# Analysis Package for the FOURTOPS Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Folder Directory](#FolderDir)
3. [Python Files](#FileDir)
4. [Closure](#ClosureFunctions)
   1. [DataLoader.py](#ClosureDataLoader)
   2. [Event.py](#ClosureEvent)
   3. [GNN.py](#ClosureGNN)
   4. [IO.py](#ClosureIO)
   5. [Models.py](#ClosureModels)
   6. [Plotting.py](#ClosurePlotting)
5. [Functions - Event - Event.py](#EventsAndCompilers)
   1. [EventVariables](#EventVariables)
   2. [Event](#Event)
   3. [EventGenerator](#EventGenerator)
6. [Functions - Event - Graphs.py](#GraphCompiler)
   1. [EventGraph](#EventGraph)
   2. [GenerateDataLoader](#GenerateDataLoader)
7. [Functions - GNN - Metrics.py](#GNNMetrics)
   1. [EvaluationMetrics](#EvaluationMetrics)
8. [Functions - GNN - Models.py](#Models)
   1. [EdgeConv](#EdgeConv)
   2. [GCN](#GCN)
   3. [InvMassGNN](#InvMassGNN)
   4. [InvMassAggr](#InvMassAggr)
9. [Functions - GNN - Optimizer.py](#OptimizerFile)
   1. [Optimizer](#Optimizer)
10. [Functions - IO - Files.py](#FilesScript)
   1. [Directories](#Directories)
   2. [WriteDirectory](#WriteDirectory)
11. [Functions - IO - IO.py](#IOScript)
   1. [File](#File)
   2. [PickleObject](#PickleObject)
   3. [UnpickleObject](#UnpickleObject)
12. [Functions - Particles - Particles.py](#ParticlesScript)
   1. [Particle](#Particle)
   2. [Lepton](#Lepton)
   3. [Electron](#Electron)
   4. [Muon](#Muon)
   5. [TruthJet](#TruthJet)
   6. [Jet](#Jet)
   7. [RCSubJet](#RCSubJet)
   8. [RCJet](#RCJet)
   9. [Top](#Top)
   10. [Truth_Top_Child](#Truth_Top_Child)
   11. [Truth_Top_Child_Init](#Truth_Top_Child_Init)
   12. [CompileParticles](#CompileParticles)
13. [Functions - Plotting - Graphs.py](#GraphsScript)
   1. [GenericAttributes](#GenericAttributes)
   2. [Graph](#Graph)
14. [Functions - Plotting - Histograms.py](#HistogramsScripts)
   1. [GenericAttributes](#GenericAttributesi_Hist)
   2. [SharedMethods](#SharedMethods)
   3. [TH1F](#TH1F)
   4. [TH2F](#TH2F)
   5. [CombineHistograms](#CombineHistograms)
   6. [CombineTGraph](#CombineTGraph)
   7. [SubfigureCanvas](#SubfigureCanvas)
   8. [TGraph](#TGraph)
15. [Functions - Tools - Alerting.py](#AlertingScripts)
   1. [Notifications](#Notifications)
   2. [Debugging](#Debugging)
16. [Functions - Tools - DataType.py](#DataTypeScript)
   1. [Threading](#Threading)
   2. [TemplateThreading](#TemplateThread)
17. [Functions - Tools - Variables.py](#VariablesScripts)
   1. [VariableManager](#VariableManager)



## Introduction <a name="introduction"></a>
This package is dedicated for a future iteration of the *Four Tops* analysis using *Graph Neural Networks*. The package expects ROOT files, that were derived from the skimmed down TOPQ DOADs, using the **BSM4topsNtuples** package (https://gitlab.cern.ch/hqt-bsm-4tops/bsm4topsntuples). In this framework, particle objects are converted into python particle objects, that are then converted into a graph data structure and subsequently converted into the **DataLoader** framework, supported by *PyTorch-Geometric*. PyTorch-Geometric is a codebase, that uses the PyTorch framework as a foundation to provide an intuitive interface to construct models, which can be applied to non euclidean data structures. 

## Folder Structure <a name="FolderDir"></a>
The codebase has the following folder structure:
- Closure : Used for testing and validating the individual functions associated with the framework.
- Functions : Container used to hold sub-functions and classes 
   - Event : Contains classes involving event generation. It reads in ROOT files and converts event objects into particle objects constituting an event 
   - GNN : Contains classes and functions relating to Graph Neural Networks. Classes include; Basic training Optimizer, GNN Model, Events to Graph, Graphs to native PyG DataLoader and Performance metrics. 
   - IO : Reads / Writes directories and does all the UpROOT reading including NumPy conversion. 
   - Particles : Definition of particles using a generic particle object that is inherited by other particle types; Jets, Leptons, Truth, etc. 
   - Plotting : Simple plotting functions for Graphs and diagnostics histograms. 
   - Tools : Some repetitive functions that are found to be useful throughout the code. 
- Plots : Container that holds plots used for diagnostics or future histograms used for presentations 
- logs : Debugging or other interesting output. 

## Closure Functions<a name="ClosureFunctions"></a>
This section will be dedicated to give a brief description of the individual closure functions, that were used to verify the individual components of the codebase. 
___
___
<a name="FileDir"></a>
###  Closure - Event.py<a name="ClosureEvent"></a>
The functions listed below are all part of a closure test involving the EventGenerator class, that converts ROOT files into Python Objects. 

```python 
def ManualUproot(dir, Tree, Branch)
```
#### Description:
This function uses the default UpRoot IO interface to open an arbitrary file directory (*dir*) and reads a given *Branch* from a *Tree*. The output is subsequently converted to the NumPy framework and returned. 
#### Input: 
- `dir`: Directory pointing to a ROOT file
- `Tree`: Tree in ROOT file 
- `Branch`: Branch in ROOT file
#### Returns:
Numpy array 

```python 
def Comparison(dir, Tree, Branch, Events, Spawned)
```
#### Description:
This function ensures that events and particle floats of the ROOT file are identical to the frameworks implementation. It throws an error if the values of events are not identical using the `assert` function.
#### Input:
- `dir`: Directory pointing to a ROOT file
- `Tree`: Tree in ROOT
- `Branch`: Branch in ROOT
- `Events`: Events to test. If `-1` then all events are tested. 
- `Spawned`: Requires an `EventGenerator` instance to be spawned with the same ROOT file. 
#### Returns:
None 

```python 
def TestParticleAssignment()
```
#### Description:
Hardcodes expected branches in the ROOT file and ensures that the `EventGenerator` reproduces outputs consistent with the reading of a normal ROOT file. This includes testing that event order is retained. Any future additions can be added and cross checked with the input ROOT file. 
#### Input:
None 
#### Returns:
None 

```python
def TestEvent()
```
#### Description:
A simple test/debug compiler that spawns events (reads ROOT Branches/Trees and assigns them to objects) and compiles a few events using a single CPU thread implementation.
#### Input:
None 
#### Output:
None 
___
___

### Closure - GNN.py<a name="ClosureGNN"></a>
The functions in this python script test anything involving Graph Neural Networks. It basically tests the interoperability between the `EventGenerator` and the PyG `DataLoader` frameworks. This is demonstrated with feature creation of edges and nodes, along with a simple training example. 

```python
def Generate_Cache()
```
#### Description:
Generates a pickled representation of the `EventGenerator` to avoid unnecessary event recompilation. The generated file can be read by the pickle class and returns the originally cached object.
#### Input:
None 
#### Output:
None - But drops files: `TruthTops` , `TruthChildren`, `TruthJets`, `Detector`, `Complete`

```python
def TestGraphObjects()
```
#### Description:
Provides test coverage for the conversion of an `Event` object, generated from the `EventGenerator` class, to a graph data representation. It uses a simple 4-top truth event and creates a completely connected graph, where nodes are tops and edges imply a relationship between top pairs. Edges and nodes have the kinematic properties eta phi and pT associated with particle objects. As closure, a plot is drawn of the graph and the expected number of edges and nodes are asserted using `assert`. 
#### Input:
None 
#### Output: 
None - But drops closure plots of the fully interconnected graph. 

```python
def TestDataImport(Compiler = "TruthTops")
```
#### Description:
This function reads a cached pickle file and tests the interfacing between the `EventGenerator` and the PyG `DataLoader` frameworks. Additionally, simple mathematical edge operations are introduced. For instance, the difference between node values are used to define edges for embedding features. Closure is achieved by creating graphs that labels edge values between nodes. 
#### Input:
`Compiler`: Any of the cached pickle functions listed under `Generate_Cache()`. 
#### Output:
None - But drops closure plots of the graphs and their labelled edge features. 

```python
def TestSimple4TopGNN()
```
#### Description:
This is a tester function that uses the PyG `DataLoader` framework and applies a very basic GNN to the given data. The input data is essentially the `FromRes` branch, which also represent the labels. This means in terms of a closure statement, the learning should converge to the input being equal to the label, since the training data is also truth. 
#### Input: 
None 
#### Output: 
None 

```python
def TestComplex4TopGNN()
```
#### Description:
A more complex closure test where particle kinematics are used as data to predict the labelling of the `FromRes` branch. This will need to be optimized and improved as part of the development phase.
#### Input:
None 
#### Output:
None 
___

### Closure - IO.py<a name="ClosureIO"></a>
A very simple closure script to check if the IO is working as expected. It tests the listing of sub-directories and files within them, along with reading multiple ROOT files at once. 

```python
def TestDir()
```
#### Description:
Reads a given directory and lists all the sub-directories along with files within them. 
#### Input:
None 
#### Output:
None 

```python
def TestIO()
```
#### Description:
Reads specified `Trees`, `Branches` and `Leaves` from ROOT files found in the given directory. 
#### Input:
None 
#### Output:
None 
___

### Closure - Plotting.py<a name="ClosurePlotting"></a>
In this closure script, various components are tested, that involve the customized histogram plotting using the `matplotlib` package. This script also checks if the mass spectrum of particles and their reconstruction chain are consistent with what is expected. 

```python
def TestTops()
```
#### Description:
In this function, a ROOT file is read into the `EventGenerator` class and only truth tops are compiled. Using their kinematic properties, their mass spectrum is calculated and should be consistent with the top mass of 172 GeV over all events. In conjunction to this, the plotting class is also tested to ensure a consistent output is achieved. An additional test is performed to verify that the top masses can be reproduced from decay products, this verifies the reconstruction of the decay chain and further validating the `EventGenerator` implementation.  
#### Input:
None 
#### Output:
None - But drops closure mass spectra for: Truth Tops, Tops from Children, Tops from Children Init, Truth Tops from Truth Jets and Detector leptons matched to children, and a comparison of anomalous truth to detector matching. 

```python
def TestResonance()
```
#### Description:
In this function, a ROOT file is read into the `EventGenerator` class and only signal tops are selected. Using their kinematic properties, the invariant mass spectra of the BSM resonance is calculated directly from the truth tops and from the decay chain. This validates the implementation of the `Particle` base class, which is used extensively in the inheritance model of the framework, and provides a generic framework for particle representations. 
#### Input:
None 
#### Output:
None - But drops closure mass spectra for BSM Resonance, that is derived from: Truth Signal tops, Signal Truth Children, Signal Children (reconstructed) from Detector Leptons and Truth Jets, and purely from Detector matched truth. Non perfectly matched detector and truth objects are also compared in plots to assess the impact of improperly matched particles. 
___

## Events and Compilers<a name="EventsAndCompilers"></a>
This section gives a detailed description of the `Event` class and associated functions used to compile ROOT events into an `EventGenerator` object. 

### Functions/Event/Event.py:
This python file is the core implementation of the `Event` and `EventGenerator` classes. In the `EventGenerator` framework, branches and trees are read on an event basis and compiled into an `Event` object that provides the truth matching and particle object compilation. 

```python
class EventVariables
```
#### Description:<a name="EventsVariables"></a>
A class that is inherited by the `EventGenerator` and used to assign string names of branches and trees for future loading purposes. It does not serve any functional purpose, except for being a convenient way to do book keeping of variables.
#### Attributes:
- `MinimalTree` : A list that contains the default `nominal` tree to read from ROOT files. Can be expanded later to include systematic `branches`. 
- `MinimalBranch` : A list of all `branches`, that are expected to be contained in the ROOT files under the specified trees.
___

```python
class Event(VariableManager, DataTypeCheck, Debugging)
```
#### Description:<a name="Event"></a>
This is a basic object class, which holds compiled particle objects and other event level attributes. It performs the main compilation of particles and aims to reconstruct the decay chain of tops in Monte Carlo samples. 
#### Init Input:
- `Debug = False` : (optional) A placeholder for analysing any issues associated with truth particle matching or any other problems, that could appear in the code. 
#### Inheritance:
- `VariableManager`: A class, which converts string variables associated with a `branch` to the appropriate values. 
- `DataTypeCheck`: A class, which keeps data structures consistent. 
- `Debugging`: A class, which contains tools that are useful for reducing redundant code commonly found in debugging.
#### Standard Output Attributes:
- `runNumber`: A default string value, which is later converted to an integer by the `VariableManager`.
- `eventNumber`: Same as above, but indicating the event number found in the ROOT file. 
- `mu`: Same as above, but represents the pile-up condition of the event. 
- `met`: Same as above, but represents the missing transverse momentum.
- `phi`: Same as above, but represents the azimuthal angle of the missing transverse momentum within the detector's reference frame. 
- `mu_actual`: Same as above, but represents the truth pile-up condition of the event. 
- `Type` : A string field indicating, that it is an `Event` object 
- `iter`: An integer value, later modified to indicate the index of the ROOT file. Used for book keeping purposes. 
- `Tree`: The `Tree` string used to fill the `Event` object. It was left as a placeholder for future systematics. 
- `TruthTops`: A dictionary containing the truth tops as particle objects.
- `TruthChildren_init`: A dictionary containing the children of the top particles, using the kinematic values associated with pre-gluon emission. 
- `TruthChildren`: A dictionary containing the children of the top particles, using the kinematic values associated with post-gluon emission. 
- `TruthJets`: A dictionary containing the truth jets in the event. 
- `Jets`: A dictionary containing the jets that were measured by the detector.
- `Muons`: A dictionary containing the muons that were measured by the detector.
- `Electrons`: A dictionary containing the electrons that were measured by the detector.
- `Anomaly`: A dictionary containing anomalous particle objects, which were not matched properly to truth particles or truth children. 
- `Anomaly_TruthMatch`: Truth objects (jets), that were not properly matched to the `TruthChildren` particle objects. 
- `Anomaly_TruthMatch_init`: Truth objects (jets), that were not properly matched to the `TruthChildren_init` particle objects. 
- `Anomaly_Detector`: Objects (jets), that were not properly matched to truth jets. 
- `BrokenEvent`: A boolean flag indicating something was not properly matched in the event. 
#### Inherited Dynamic Attributes:
- `Branches`: An empty list that is used by the `VariableManager`.
- `KeyMap`: An empty dictionary used to map relevant branches found in ROOT files to object variables of the event. The idea is to update the string representation of an object attribute to the value found in a ROOT file (e.g. runNumber (string) -> runNumber (value in ROOT file)). 
- `Debug`: A boolean trigger, that is used as a placeholder to inspect the object.
#### Class Implemented Functions: 

```python 
def ParticleProxy(self, File)
```
##### Description: 
A function, which routes a given branch variable to the associated object and assigns the object attribute the value of the routed branch (i.e. branch to variable assignment). The `File` argument requires a `File` object to be given.
##### Affected Internal Variables: 
- `BrokenEvent`: is set to `True` if an error occurs during value reading (caused by missing branch etc.). 
- `Branches`: The list is updated via an inherited function called `ListAttributes`. This function updates the branch strings expected to be found in a specified ROOT file.
- `TruthJets`: A dictionary used to map ROOT truthjet branch strings with associated values. 
- `Jets`: A dictionary used to map ROOT jet branch strings with associated values. 
- `Electrons`: A dictionary used to map ROOT electron branch strings with associated values. 
- `Muons`: A dictionary used to map ROOT muons branch strings with associated values. 
- `TruthChildren`: A dictionary used to map ROOT truth_children branch strings with associated values. 
- `TruthChildren_init`: A dictionary used to map ROOT truth_children_init branch strings with associated values. 
- `TruthTops`: A dictionary used to map ROOT truth_top branch strings with associated values. 
- `runNumber`, `eventNumber`, `mu`, `met`, `mu_actual`, `phi`, `iter`: String values are updated with ROOT and other values. 

```python
def MatchingRule(self, P1, P2, dR)
```
##### Description: 
Two particles are matched according to certain matching rules (See implementation for details) and their closest dR = sqrt(dphi^2 - deta^2). 
##### Output:
`True`, `False`

```python
def DeltaRMatrix(self, List1, List2)
```
##### Description:
Calculates the dR matrix between a list of particle objects. 
##### Affected Internal Variables:
- `dRMatrix`: (Newly Spawned) A sorted list with smaller dR pairs placed first - ```python [L_i2, L_i1, dR]```. 

```python
def DeltaRLoop(self)
```
##### Description:
Iterates through the `dRMatrix` variable and applies matching rules. Particles matched are appended to their parent particle under the `Decay_init` and `Decay` lists (part of the generic `Particle` class).
##### Affected Internal Variables:
- `Anomaly`: Dictionary is filled according to which matching was performed. 
- `Anomaly_Detector`, `Anomaly_TruthMatch`, `Anomaly_TruthMatch_init`: Are set to `True` if applicable. 

```python
def CompileSpecificParticles(self, particles = False)
```
##### Description: 
Performs matching for specific particle pairs, such as tops to their truth children objects and compiles all event particle objects. 
##### Input:
- `particles = False`: Allows for the following string arguments; `TruthTops` (Compiles only tops), `TruthChildren` (Compiles truth children and matches them to their respective tops), `TruthJets` (Only compiles truth jet objects), `Detector` (Compiles detector objects).
##### Affected Internal Variables:
- `TruthChildren`, `TruthChildren_init`, `TruthTops`, `TruthJets`, `Jets`, `Muons`, `Electrons`: Dictionaries are converted to lists containing the particle objects.

```python
def CompileEvent(self)
```
##### Description:
Compiles the event and performs the matching of particles to their parents. It aims to construct the original decay chain of the individual tops contained in the event. 
##### Affected Internal Variables:
- `TruthChildren`, `TruthChildren_init`, `TruthTops`, `TruthJets`, `Jets`, `Muons`, `Electrons`: Dictionaries are converted to lists containing the particle objects.

```python
def DetectorMatchingEngine(self)
```
##### Description:
Matches detector particles with truth jets and leptons to the truth children. This function is called from the main compiler routine. 
##### Affected Internal Variables:
- `CallLoop`: A string that is updated internally according to which matching engine is being called. 

```python
def TruthMatchingEngine(self)
```
##### Description:
Matches truth jet and lepton particles with truth children. This function is called from the main compiler routine. 
##### Affected Internal Variables:
- `CallLoop`: A string that is updated internally according to which matching engine is being called. 
___

```python
class EventGenerator(UpROOT_Reader, Debugging, EventVariables)
```
#### Description:<a name="EventGenerator"></a>
A framework used to represent the contents of a ROOT file as python objects. It does so, by generating individual `Event` objects and passes the event specific values from branches to the object. Furthermore, it also contains a multithreading function used to make the compilation more efficient. 
#### Init Input: 
- `dir`: (Required) A directory path to ROOT files. This path can be a specific ROOT file or a directory containing multiple ROOT files. 
- `Verbose`: (Optional) Set to `True` by default. Enables or disables notifications and debugging messages.
- `DebugThres`: (Optional) Set to `-1` by default. It is used to specify the number of events to loop over. The default value implies all events. 
- `Debug = False` : (optional) A placeholder for analysing any issues associated with truth particle matching or any other problems, that could appear in the code. 
#### Inheritance: 
- `UpROOT_Reader`: A class used to generate `File` objects and checks for ROOT files within the given directory. 
- `Debugging`: A class, which contains tools that are useful for reducing redundant code commonly found in debugging.
- `EventVariables`: A class containing lists of trees and branches, which are to be included in the compilation process. 
#### Standard Output Attributes:
- `Events`: A dictionary containing the compiled `Event` objects.
#### Inherited Dynamic Attributes:
- `Caller`: A string indicating the function being called. 
- `FileObjects`: A dictionary of ROOT files mapped to a directory. 
- `Debug`: A boolean flag, that disables the verbosity. 
- `MinimalTree`: A list containing the string of trees, that the generator should read and compile events from.
- `MinimalBranch`: A list containing the string of branches, that the generator should read and compile events from.
#### Class Implemented Functions:

```python
def SpawnEvents(self, Full = False)
```
##### Description:
A function, which iterates through `FileObjects`, that contain individual ROOT files and generates `Event` objects. These objects are filled with basic event attributes and mapped to the associated ROOT filename. The `Full` argument is a boolean flag, which toggles whether to use *all* trees in the ROOT files or only include the nominal tree. By default this variable is set to `False`, indicating the use of only the nominal tree. 
##### Affected Internal Variables:
- `Events`: The dictionary is populated with uncompiled `Event` objects, but mapped to the associated ROOT file path. 
- `FileObjects`: The dictionary is cleared to reduce RAM usage. 
##### Output:
None 

```python 
def CompileEvent(self, SingleThread = False, particle = False)
```
##### Description:
A function, which iterates through the `Events` dictionary and batches the `Event` objects into N partitions, that are submitted to the multithreading compiler. The `SingleThread` flag toggles the use of batching for multithreading. If this variable is set to `True`, only a single thread is used to compile the events. The `particle` flag is used to denote which of particles should be compiled. 
##### Affected Internal Variables:
`Batches`: (Newly Spawned) A dictionary, which maps the CPU threads to the batches. 
`Events`: The uncompiled events are updated to fully compiled events. 
`Caller`: The string of the invoked function is updated to `EVENTCOMPILER` and is used for the notification prompt. 
##### Output:
None 
___

## Graph Neural Network Models and Optimizer<a name="GNNsOptimizer"></a>
This section gives a detailed description of the Graph Neural Network models that are implemented and the associated optimizer. This file should be used to implement different GNNs. 

### Functions/GNN/GNN.py
This file is dedicated for GNN model development. This section will be rather quickly outdated since the GNN implementation will change over time and new models are tested. However, this file contains the generic optimizer framework and will most likely contain performance metrics. 

```python
class EdgeConv(MessagePassing)
```
#### Description:<a name="EdgeConv"></a>
This is an implementation of a very basic GNN model. This model will be extended and altered over time. 
#### Init Input:
- `in_channels` : An integer value that represents the number of inputs to the MLP. 
- `out_channels`: An integer value indicating the output decisions, i.e. labels. 
#### Inheritance:
- `MessagePassing`: A class originating from the PyGeometric package to perform graph message passing (https://towardsdatascience.com/introduction-to-message-passing-neural-networks-e670dc103a87). 
#### Attributes:
- `mlp` : A multi-layer perceptron initializer that maps input and output channel dimensions. 
___

```python
class Optimizer
```
##### Description:<a name="Optimizer"></a>
A custom implementation of the learning optimizer used to minimize the model parameters of a GNN. 
#### Init Input:
None 
#### Inheritance:
None 
#### Standard Attributes:
- `Optimizer` : A placeholder for the optimizer used to minimize the parameters of the GNN. This can be adjusted as needed. 
- `Model` : A placeholder that is assigned once a model function is activated. 
- `DataLoader` : The native graph data object of the PyGeometric framework. This can be generated from the `GenerateDataLoader` class. 
- `Epochs` : The number of learning epochs that the GNN is going to iterate through. By default this is set to 100. 
- `LearningRate` : The learning rate of the GNN. By default this is set to 0.01.
- `WeightDecay` : The decay rate of the weight during the learning. By default this is set to 1e-6. 
- `Device` : An automatic switch (can be overwritten), that chooses a CUDA device is available. Else it uses simply the CPU. 
#### Class Implemented Functions: 

```python
def Learning(self)
```
##### Description:
Trains the given `Model` attribute using the defined `Optimizer` attribute, via the CrossEntropyLoss method. 
##### Output: 
None 

```python
def EpochLoop(self)
```
##### Description:
Runs over the given `DataLoader` over a number of `Epochs`. For this function to run, a `Model` implementation needs to be given or defined. 
##### Affected Internal Variables:
- `data` : Updates the current data object that will be used for training. 
##### Output:
None 

```python
def Prediction(self, in_channel, output_dim)
``` 
##### Description:
A function that outputs classification prediction made by the trained model given input channels of the data. 
##### Output:
`y_p` : A prediction vector of the labels. 

```python
def DefineEdgeConv
``` 
##### Description:
This function sets the `Model` and `Optimizer` attributes of the class and interfaces the `EdgeConv` class with the learning framework. 
##### Affected Internal Variables:
- `Model` : Defines the current model to train. 
- `Optimizer` : Defines the current optimizer to be used for the parameter minimization. 
##### Output:
None 
___

### Functions/GNN/Graphs.py<a name="Graphs"></a>
This file contains classes and functions that convert the given `EventGenerator` object into event graphs that are subsequently parsed into the PyG `DataLoader` framework. Event graphs can also be modified to include feature embedding for the nodes and edges. By default some basic feature embeddings are defined and can be extended easily within the `CreateEventGraph` class.

```python
class CreateEventGraph
```
#### Description: <a name="CreateEventGraph"></a>
An object class that transforms particle objects and their attributes into event graphs. These graphs can contain any arbitrary number of features and connections.
#### Init Input:
- `Event` : An instance of the `Event` object, that was generated through the `EventGenerator` class. 
#### Attributes:
- `G`: The generated event graph. By default this graph is empty and defined by the `NetworkX` package. 
- `Truth_G`: This is an optional graph, which defines the truth connection between particle nodes and lists all truth particles, that may have not been reconstructed in the detector. 
- `EdgeAttributes`: An empty list used to append names of available edge features. 
- `NodeAttributes`: An empty list used to append names of available node features. 
- `ExcludeSelfLoops`: A boolean flag to indicate whether nodes should contain an edge associated with themselves. By default this is set to `True`.
- `DefaultEdgeWeight`: A default float indicating the connection strength between nodes, i.e. the edge strength. 
#### Class Implemented Functions:

```python
def CreateParticleNodes(self)
```
##### Description:
A function spawning nodes on the graph attribute `G`. 
##### Affected Internal Variables:
- `Nodes`: (Newly Spwaned) A list of integers representing the node index. 
##### Output:
None 

```python
def CreateParticlesEdgesAll(self)
```
##### Description:
Creates a fully connected graph from the defined nodes. 
##### Affected Internal Variables:
- `Edges`: Records the index pairs of nodes that are connected. 
##### Output:
None 

```python
def CreateDefaultEdgeWeights(self)
```
##### Description:
Uses the recorded edge pairs and assigns the "weight" feature to the graph edges with the value defined in `DefaultEdgeWeight`.
##### Output:
None 

```python
def CalculateEdgeAttributes(self, fx)
```
##### Description:
Given some function `fx`, the relation feature between two nodes is calculated and assigned to the edges.
##### Affected Internal Variables:
- `EdgeAttributes`: Features names are appended and assigned a `d_<attribute>`. 
##### Output:
None 

```python
def CalculateNodeAttributes(self)
```
##### Description:
Iterates through all given attributes of the node (given by the user) and embeds it to the graph. 
##### Output:
None 

```python
def CalculateNodeDifference(self)
```
##### Description:
Calculates the difference between a given attribute of the nodes. This function is called through the `CalculationProxy` function. 
##### Affected Internal Variables:
- `EdgeAttributes` : The name of the operation is recorded as a string. 
##### Output:
None 

```python
def CalculateNodeMultiplication(self)
```
##### Description:
Multiplies a given attribute of two nodes with each other. 
##### Affected Internal Variables:
- `EdgeAttributes` : The name of the operation is recorded as a string. 
##### Output:
None 

```python
def CalculateParticledR(self)
```
##### Description: 
Calculates the radial distance between two nodes (particles).
##### Affected Internal Variables:
- `EdgeAttributes` : The name of the operation is recorded as a string.
##### Output:
None 

```python
def CalculateNodeMultiplicationIndex(self)
```
##### Description:
Multiplies the `Index` attribute of the nodes (from the generic particle class) with each other and assigns a value of `-1` if the indexes do not match. 
##### Affected Internal Variables:
- `EdgeAttributes` : The name of the operation is recorded as a string.
##### Output:
None 

```python
def CalculateInvariantMass(self)
```
##### Description:
Uses the kinematic properties of nodes to define a four vector that is summed and used to calculate the invariant mass. 
##### Affected Internal Variables:
- `EdgeAttributes` : The name of the operation is recorded as a string. 
##### Output:
None 

```python 
def CalculationProxy(self, Dict)
```
##### Description:
Uses a dictionary to route the function that should be applied to an attribute. Specific keywords are used to trigger the Calculation functions. Some defined keywords are; "Diff", "Multi", "MultiIndex", "dR", "" and "invMass". 
##### Affected Internal Variables:
- `EdgeAttributes` : The name of the operation is recorded as a string.
- `NodeAttributes` : Records the name of the attribute of the nodes that is used for the calculation. 
##### Output:
None 

```python
def ConvertToData(self)
```
##### Description:
Converts the compiled NetworkX graph into a Data object that is later used in the PyG `DataLoader` framework. 
##### Affected Internal Variables:
None 
##### Output:
- `torch_geometric.data.Data` : An object used for the `DataLoader` framework. 
___

```python 
class GenerateDataLoader 
```
#### Description:<a name="GenerateDataLoader"></a>
A class used to interface the `EventGenerator` with the `DataLoader` framework by converting individual event objects into graph objects, that can be subsequently imported to the GNN framework.
#### Init Input:
- `Bundle`: An `EventGenerator` instance that should be used. 
#### Inheritance:
None 
#### Standard Output Attributes
- `ExcludeAnomalies`: A boolean flag that indicates to skip events where particles are not well matched to truth particles. By default this is set to `False`.
- `Device` : An automatic switch (can be overwritten), that chooses a CUDA device is available. Else it uses simply the CPU. 
- `EdgeAttributes`: An empty dictionary used to define the attributes and the calculation that should be applied to them. 
- `NodeAttributes`: An empty dictionary used to deine the attriubutes of the nodes. 
- `DataLoader` : Returns the end result of compiler. If successful this would return the `DataLoader` object used by PyG. 
- `DefaultBatchSize`: Defines the number of graphs that are disconnected and batched for the learning. By default this is set to 20. 
#### Class Implemented Functions:

```python
def GetEventParticles(self, Ev, Branch)
```
##### Description:
Collects the requested compiled particle list (e.g. TruthTops, Detector, TruthChildren) from the given `Event` object. 
##### Affected Internal Variables:
None 
##### Output:
- `False`: The given `Event` does not have a compiled version of the particles for the specified branch. 
- `list(Particles)` : A list of particle objects.

```python
def CreateEventData(self, Event)
```
##### Description:
Converts the given `Event` object into an event graph and concatenates graph attributes to a single `Data` object. 
##### Affected Internal Variables:
None 
##### Output:
`torch_geometric.data.Data` : A modified object which has additional attributes, "x" and "edge_attr" used as input to the GNN model. 


