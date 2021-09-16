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
This package is dedicated for a future iteration of the *Four Tops* analysis using *Graph Neural Networks*. The package expects ROOT files, that are derived from skimmed down TOPQ DOADs, using the **BSM4topsNtuples** package (https://gitlab.cern.ch/hqt-bsm-4tops/bsm4topsntuples). In this framework, particle objects are converted into python particle object, that are then converted into a graph data structure and subsequently converted into the **DataLoader** framework, supported by *PyTorch-Geometric*. PyTorch-Geometric is a codebase, that uses the PyTorch framework as a foundation to provide an intuitive interface to construct models, which can be applied to non euclidean data structures. 

## Folder Structure <a name="FolderDir"></a>
The codebase has the following folder structure:
- Closure : Used for testing and validating the individual functions associated with the framework.
- Functions : Container used to hold sub-functions and classes 
   - Event : Contains classes involving event generation. It reads in ROOT files and converts event objects into particles objects constituting an event 
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
This section will be dedicated to give a brief description of what functions are in which files and how they are interfaced. 

###  Closure - Event.py<a name="ClosureEvent"></a>
The functions listed below are all part of a closure surrounding the EventGenerator class, that converts ROOT files into Python Objects. 
___

```python 
def ManualUproot(dir, Tree, Branch)
```
#### Description:
This function uses the default UpRoot IO interface to open an arbitrary file directory (*dir*) and reads a given *Branch* from a *Tree*. The output is subsequently converted to the Numpy framework and returned. 

#### Input: 
`dir`: Directory pointing to a ROOT file

`Tree`: Tree in ROOT file 

`Branch`: Branch in ROOT file

#### Returns:
Numpy array 

```python 
def Comparison(dir, Tree, Branch, Events, Spawned)
```

#### Description:
This function ensures that the events and the particle floats of the ROOT file are identical to the frameworks implementation. It throws an error if the values of events are not identical through the `assert` function.

#### Input:
`dir`: Directory pointing to a ROOT file

`Tree`: Tree in ROOT

`Branch`: Branch in ROOT

`Events`: Events to test. If `-1` then all events are tested. 

`Spawned`: Requires an `EventGenerator` instance to be spawned with the same ROOT file. 

#### Returns:
None 

```python 
def TestParticleAssignment()
```

#### Description:
Hardcodes all the branches in the ROOT file and ensures that the `EventGenerator` framework produces output consistent with that expected with reading a normal ROOT file. This includes testing that event order is retained. Any future additions can be added and cross checked with the input ROOT file. 

#### Input:
None 

#### Returns:
None 

```python
def TestEvent()
```

#### Description:
A simple test/debug compiler that spawns events (reads ROOT Branches/Trees and assigns them to objects) and compiles a few events using a single thread implementation.

#### Input:
None 

#### Output:
None 

### Closure - GNN.py<a name="ClosureGNN"></a>
The functions in this python script test anything involving Graph Neural Networks. It basically tests the interoperability between the `EventGenerator` and the PyG `DataLoader` frameworks. This is demonstrated with feature creation of edges and nodes, along with a simple training example. 
___
```python
def Generate_Cache()
```

#### Description:
Generates a pickle representation of the `EventGenerator` to avoid unnecessary event recompilation. The generated file can be read by the pickle framework and returns the cached object for latent use.

#### Input:
None 

#### Output:
None - But drops files: `TruthTops` , `TruthChildren`, `TruthJets`, `Detector`, `Complete`

```python
def TestGraphObjects()
```

#### Description:
Provides test coverage for the conversion of an `Event` object, generated from the `EventGenerator` class, to a graph data representation. It uses a simple 4-top truth event and creates a completely connected graph, where the nodes are tops and the edges imply a relationship between top pairs. The edges and nodes have the kinematic properties eta phi and pT associated with particle objects. As closure, a plot is drawn of the graph and the expected number of edges and nodes are asserted using `assert`. 

#### Input:
None 

#### Output: 
None - But drops closure plots of the fully interconnected graph. 

```python
def TestDataImport(Compiler = "TruthTops")
```

#### Description:
This function reads in a cached pickle file and tests the interfacing between the `EventGenerator` and the PyG `DataLoader` frameworks. Additionally, simple mathematical edge operations are introduced. For instance the difference between node values defining the edges between nodes, such that edge feature embedding is possible. Closure is achieved by creating graphs that labels edge values between nodes. 

#### Input:
`Compiler`: Any of the cached pickle functions listed under `Generate_Cache()`. 

#### Output:
None - But drops closure plots of the graphs and their labelled edge features. 

```python
def TestSimple4TopGNN()
```

#### Description:
This is a tester function that uses the PyG `DataLoader` framework and applies a very basic GNN to the given data. The input data is essentially the `FromRes` branch, which is also the underlying labels. This means in terms of a closure statement, the learning should converge to the input being equal to the label, since the training data is also truth. 

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

### Closure - IO.py<a name="ClosureIO"></a>
A very simple closure script check if the IO is working as expected. It tests the listing of sub-directories and files within them, along with reading multiple ROOT files at once. 
___
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

### Closure - Plotting.py<a name="ClosurePlotting"></a>
In this closure script, various components are tested, that involve the plotting capabilities with `matplotlib`. This script also serves the purpose to check if the mass spectra of particles and their reconstruction chain are consistent with what is expected. 
___
```python
def TestTops()
```

#### Description:
In this function, a ROOT file is read into the `EventGenerator` framework and only truth tops are selected. Using their kinematic properties, their mass spectra is inferred and should be consistent with 172 GeV over all events. In conjunction, the plotting code class is also tested to ensure future scalability of the project. An additional test is performed to verify that the top masses can be reproduced from decay products to verify the reconstruction of the decay chain, further validating the `EventGenerator` implementation.  

#### Input:
None 

#### Output:
None - But drops closure mass spectra for: Truth Tops, Tops from Children, Tops from Children INIT, Truth Tops from Truth Jets and Detector leptons matched to children, and a comparison of anomalous truth to detector matching. 

```python
def TestResonance()
```

#### Description:
In this function, a ROOT file is read into the `EventGenerator` framework and only signal tops are selected. Using their kinematic properties, the mass spectra of the BSM resonance is inferred from the tops directly and from the decay chain. This validates the implementation of the `Particle` base class, which is used extensively in the inheritance model of the ROOT objects, to provide a generic framework for particle representations. 

#### Input:
None 

#### Output:
None - But drops closure mass spectra for BSM Resonance, that is derived from: Truth Signal tops, Signal Truth Children, Signal Children (reconstructed) from Detector Leptons and Truth Jets, and purely from Detector matched truth. Non perfectly matched detector and truth objects are also compared in plots to assess the impact of improperly matched particles. 

### Functions - Event - Event.py:
This python file is the core implementation of the `Event` and `EventGenerator` classes. In the `EventGenerator` framework, branches and trees are read on a event basis and compiled into an `Event` object that provides the truth matching and particle object compilation. 
___

```python
class EventVariables
```
#### Description:
A class that is inherited by `EventGenerator` to define all needed branches and trees to enable events to be compiled appropriately. It does not serve any functional purpose, except for being a convenient way to do book keeping of variables.

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
- `VariableManager`: A class, which assigns string based variables associated with a `branch` to a value. 
- `DataTypeCheck`: A class, which keeps data structures consistent. 
- `Debugging`: A class, which contains tools that are quite useful for debugging purposes and don't need to be rewritten multiple times in the codebase.

#### Standard Output Attributes:
- `runNumber`: A default string value, that is later converted to an integer by the `VariableManager`.
- `eventNumber`: Same as above, but indicating the event number found in the ROOT file. 
- `mu`: Same as above, but represents the pile-up condition of the event. 
- `met`: Same as above, but represents the missing transverse momentum.
- `phi`: Same as above, but represent the azimuthal angle of, that the missing transverse momentum is pointing to in the detector's reference frame. 
- `mu_actual`: Same as above, but represents the truth pile-up condition of the event. 
- `Type` : A string field indicating, that it is an `Event` object 
- `iter`: An integer value, that is later modified to indicate the index of the ROOT file. This is used for bookkeeping purposes. 
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
- `KeyMap`: An empty dictionary used to match string `Branch` contained in the ROOT file to update the variable of the event (e.g. runNumber (string) -> runNumber (value in ROOT file)). 
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
`True, False`

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


