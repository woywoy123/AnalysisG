# Analysis Package for the FOURTOPS Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Folder Directory](#FolderDir)
3. [Python Files](#FileDir)
4. [Functions and Classes](#FunctionClasses)
   1. [Closure - Event.py](#ClosureEvent)
   2. [Closure - GNN.py](#ClosureGNN)
   3. [Closure - IO.py](#ClosureIO)
   4. [Closure - Plots.py](#ClosurePlotting)

## Introduction <a name="introduction"></a>
This package is dedicated for a future iteration of the *Four Tops* analysis using *Graph Neural Networks*. The package expects ROOT files, that are derived from skimmed down TOPQ DOADs, using the **BSM4topsNtuples** package (https://gitlab.cern.ch/hqt-bsm-4tops/bsm4topsntuples). In this framework, particle objects are converted into python particle objects, that are then converted into a graph data structure and subsequently converted into the **DataLoader** framework, supported by *PyTorch-Geometric*. PyTorch-Geometric is a codebase, that uses the PyTorch framework as a foundation to provide an intuitive interface to construct models, which can be applied to non euclidean data structures. 

## Folder Structure <a name="FolderDir"></a>
The codebase has the following folder structure:
- Closure : Used for testing and validating the individual functions associated with the framework.
- Functions : Container used to hold sub-functions and classes 
   - Event : Contains classes involving event generation. It reads in ROOT files and converts event objects into particle objects constituting an event 
   - GNN : Contains classes and functions relating to Graph Neural Networks. Classes include; Basic training Optimizer, GNN Model, Events to Graph, Graphs to native PyG DataLoader and Performance metrics. 
   - IO : Reads / Writes directories and does all the UpROOT reading including NUMPY conversion. 
   - Particles : Definition of particles using a generic particle object that is inherited by other particle types; Jets, Leptons, Truth, etc. 
   - Plotting : Simple plotting functions for Graphs and diagnostics histograms. 
   - Tools : Some repetitive functions that are found to be useful throughout the code. 
- Plots : Container that holds plots used for diagnostics or future histograms used for presentations 
- logs : Debugging or other interesting output. 

## Python File Index<a name="FileDir"></a>
- Closure 
   - Event.py
   - GNN.py
   - IO.py
   - Plotting.py
- Functions
   - Event
      - Event.py
   - GNN
      - GNN.py
      - Graphs.py
      - Metrics.py
   - IO
      - Files.py
      - IO.py
   - Particles
      - Particles.py
   - Plotting 
      - Graphs.py
      - Histograms.py
    - Tools
      - Alerting.py
      - DataTypes.py
      - Variables.py
   - GNN
      - GNN.py
      - Graphs.py
      - Metrics.py
   - IO
      - Files.py
      - IO.py
   - Particles
      - Particles.py
   - Plotting 
      - Graphs.py
      - Histograms.py
   - Tools
      - Alerting.py
      - DataTypes.py
      - Variables.py

## Python Functions and Classes<a name="FunctionClasses"></a>
This section will be dedicated to give a brief description of what functions are contained in files and how they are interfaced. 
___
___

###  Closure - Event.py<a name="ClosureEvent"></a>
The functions listed below are all part of a closure test involving the EventGenerator class, that converts ROOT files into Python Objects. 

```python 
def ManualUproot(dir, Tree, Branch)
```
#### Description:
This function uses the default UpRoot IO interface to open an arbitrary file directory (*dir*) and reads a given *Branch* from a *Tree*. The output is subsequently converted to the Numpy framework and returned. 
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
Hardcodes all the branches in the ROOT file and ensures that the `EventGenerator` reproduces outputs consistent with the reading of a normal ROOT file. This includes testing that event order is retained. Any future additions can be added and cross checked with the input ROOT file. 
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
None - But drops closure mass spectra for: Truth Tops, Tops from Children, Tops from Children INIT, Truth Tops from Truth Jets and Detector leptons matched to children, and a comparison of anomalous truth to detector matching. 

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
___

### Functions - Event - Event.py:
This python file is the core implementation of the `Event` and `EventGenerator` classes. In the `EventGenerator` framework, branches and trees are read on an event basis and compiled into an `Event` object that provides the truth matching and particle object compilation. 

```python
class EventVariables
```
#### Description:
A class that is inherited by `EventGenerator`, that uses the string names of branches and trees as a way to mark and load them later on. It does not serve any functional purpose, except for being a convenient way to do book keeping of variables.
#### Attributes:
- `MinimalTree` : A list that contains the default `nominal` tree to read from ROOT files. Can be expanded later to include systematic `branches`. 
- `MinimalBranch` : A list of all `branches`, that are expected to be contained in the ROOT files.
___

```python
class Event(VariableManager, DataTypeCheck, Debugging)
```
#### Init Input:
- `Debug = False` : (optional) A placeholder for analysing any issues associated with truth particle matching or any other problems in the code. 
#### Inheritance:
- `VariableManager`: A class, which converts string variables associated with a `branch` to the appropriate values. 
- `DataTypeCheck`: A class, which keeps data structures consistent. 
- `Debugging`: A class, which contains tools that are quite useful for debugging purposes and reduce redudant code in the codebase.
#### Standard Output Attributes:
- `runNumber`: A default string value that is later converted to an integer by the `VariableManager`.
- `eventNumber`: Same as above, but indicating the event number found in the ROOT file. 
- `mu`: Same as above, but represents the pile-up condition of the event. 
- `met`: Same as above, but represents the missing transverse momentum.
- `phi`: Same as above, but represents the azimuthal angle of the missing transverse momentum pointing to in the detector's reference frame. 
- `mu_actual`: Same as above, but represents the truth pile-up condition of the event. 
- `Type` : A string field indicating, that it is an `Event` object 
- `iter`: An integer value, that is later modified to indicate the index of the ROOT file. This is used for book keeping purposes. 
- `Tree`: The `Tree` string used to fill the `Event` object. It was left as a placeholder for future systematics. 
- `TruthTops`: A dictionary containing the truth tops as particle objects.
- `TruthChildren_init`: A dictionary containing the children of the top particles, but inheriting the kinematic values associated with pre-gluon emission. 
- `TruthChildren`: A dictionary containing the children of the top particles, but inheriting the kinematic values associated with post-gluon emission. 
- `TruthJets`: A dictionary containing the truth jets in the event. 
- `Jets`: A dictionary containing the jets that were measured by the detector.
- `Muons`: A dictionary containing the muons that were measured by the detector.
- `Electrons`: A dictionary containing the electrons that were measured by the detector.
- `Anomaly`: A dictionary containing anomalous particle objects, that did not match properly to truth particles or truth children. 
- `Anomaly_TruthMatch`: Truth objects (jets), that were not well matched to the `TruthChildren` particle objects. 
- `Anomaly_TruthMatch_init`: Truth objects (jets), that were not well matched to the `TruthChildren_init` particle objects. 
- `Anomaly_Detector`: Objects (jets), that were not well matched to truth jets. 
- `BrokenEvent`: A boolean flag indicating something was not well matched in the event. 
#### Inherited Dynamic Attributes:
- `Branches`: An empty list that is used by the `VariableManager`.
- `KeyMap`: An empty dictionary used to match the `Branch` string contained in the ROOT file to update the variable of the event (e.g. runNumber (string) -> runNumber (value in ROOT file)). 
- `Debug`: A boolean trigger, that is used as a placeholder to inspect the object.
#### Class Implemented Functions: 

```python 
def ParticleProxy(self, File)
```
##### Description: 
A function that creates a string to variable mapping given a file object, that contains an opened ROOT file (i.e. branch to variable assignment). 
##### Affected Internal Variables: 
- `BrokenEvent`: is set to `True` if an error occurs during value reading (caused by missing branch etc.). 
- `Branches`: State change of branch strings expected in given ROOT file are saved in a list used for later compilation. 
- `TruthJets`: Dictionaries are used to map ROOT truthjet branch strings and their values. 
- `Jets`: Dictionaries are used to map ROOT jet branch strings and their values. 
- `Electrons`: Dictionaries are used to map ROOT electron branch strings and their values.
- `Muons`: Dictionaries are used to map ROOT muons branch strings and their values.
- `TruthChildren`: Dictionaries are used to map ROOT truth_children branch strings and their values.
- `TruthChildren_init`: Dictionaries are used to map ROOT truth_children_init branch strings and their values.
- `TruthTops`: Dictionaries are used to map ROOT truth_top branch strings and their values.
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
- `dRMatrix`: (Newly Spwaned) Sorted list with smaller dR pairs placed first - [L_i2, L_i1, dR]. 

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
- `TruthTops`: Compiles only tops.
- `TruthChildren`: Compiles truth children and matches them to their respective tops.
- `TruthJets`: Compiles truth jet objects, no matching is performed at this stage. 
- `Detector`: Compiles detector objects, no matching is performed at this stage.
##### Affected Internal Variables:
- `TruthChildren`, `TruthChildren_init`, `TruthTops`, `TruthJets`, `Jets`, `Muons`, `Electrons`: Dictionaries are converted to lists containing the particle objects.

```python
def CompileEvent(self)
```
##### Description:
Compiles the event and performs all the matching of particles. 
##### Affected Internal Variables:
- `TruthChildren`, `TruthChildren_init`, `TruthTops`, `TruthJets`, `Jets`, `Muons`, `Electrons`: Dictionaries are converted to lists containing the particle objects.

```python
def DetectorMatchingEngine(self)
```
##### Description:
Matches detector particles with truth jets and leptons from truth children. This function is called from the main compiler routine. 
##### Affected Internal Variables:
- `CallLoop`: A string that is updated internally according to which matching engine is being used. 

```python
def TruthMatchingEngine(self)
```
##### Description:
Matches truth jet and lepton particles with truth children. This function is called from the main compiler routine. 
##### Affected Internal Variables:
- `CallLoop`: A string that is updated internally according to which matching engine is being used. 
___
___

### Functions - GNN - GNN.py
This file is dedicated for GNN model development. This section will be rather quickly outdated since the GNN implementation will change over time and new models are tested. However, this file contains the generic optimizer framework and will most likely contain performance metrics. 

```python
class EdgeConv(MessagePassing)
```
#### Description:
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
##### Description:
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
___

### Functions - GNN - Graphs.py
This file contains classes and functions that convert the given `EventGenerator` object into event graphs that are subsequently parsed into the PyG `DataLoader` framework. Event graphs can also be modified to include feature embedding for the nodes and edges. By default some basic feature embeddings are defined and can be extended easily within the `CreateEventGraph` class.

```python
class CreateEventGraph
```
#### Description: 
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
CalculateEdgeAttributes(self, fx)
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
#### Description:
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


