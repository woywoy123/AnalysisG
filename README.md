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
ManualUproot(dir, Tree, Branch)
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
Comparison(dir, Tree, Branch, Events, Spawned)
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
TestParticleAssignment()
```

#### Description:
Hardcodes all the branches in the ROOT file and ensures that the `EventGenerator` framework produces output consistent with that expected with reading a normal ROOT file. This includes testing that event order is retained. Any future additions can be added and cross checked with the input ROOT file. 

#### Input:
None 

#### Returns:
None 

```python
TestEvent()
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
Generate_Cache()
```

#### Description:
Generates a pickle representation of the `EventGenerator` to avoid unnecessary event recompilation. The generated file can be read by the pickle framework and returns the cached object for latent use.

#### Input:
None 

#### Output:
None - But drops files: `TruthTops` , `TruthChildren`, `TruthJets`, `Detector`, `Complete`
___
```python
TestGraphObjects()
```

#### Description:
Provides test coverage for the conversion of an `Event` object, generated from the `EventGenerator` class, to a graph data representation. It uses a simple 4-top truth event and creates a completely connected graph, where the nodes are tops and the edges imply a relationship between top pairs. The edges and nodes have the kinematic properties eta phi and pT associated with particle objects. As closure, a plot is drawn of the graph and the expected number of edges and nodes are asserted using `assert`. 

#### Input:
None 

#### Output: 
None - But drops closure plots of the fully interconnected graph. 
___
```python
TestDataImport(Compiler = "TruthTops")
```

#### Description:
This function reads in a cached pickle file and tests the interfacing between the `EventGenerator` and the PyG `DataLoader` frameworks. Additionally, simple mathematical edge operations are introduced. For instance the difference between node values defining the edges between nodes, such that edge feature embedding is possible. Closure is achieved by creating graphs that labels edge values between nodes. 

#### Input:
`Compiler`: Any of the cached pickle functions listed under `Generate_Cache()`. 

#### Output:
None - But drops closure plots of the graphs and their labelled edge features. 
___
```python
TestSimple4TopGNN()
```

#### Description:
This is a tester function that uses the PyG `DataLoader` framework and applies a very basic GNN to the given data. The input data is essentially the `FromRes` branch, which is also the underlying labels. This means in terms of a closure statement, the learning should converge to the input being equal to the label, since the training data is also truth. 

#### Input: 
None 

#### Output: 
None 
___
```python
TestComplex4TopGNN()
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
TestDir()
```

#### Description:
Reads a given directory and lists all the sub-directories along with files within them. 

#### Input:
None 

#### Output:
None 
___
```python
TestIO()
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
TestTops()
```

#### Description:
In this function, a ROOT file is read into the `EventGenerator` framework and only truth tops are selected. Using their kinematic properties, their mass spectra is inferred and should be consistent with 172 GeV over all events. In conjunction, the plotting code class is also tested to ensure future scalability of the project. An additional test is performed to verify that the top masses can be reproduced from decay products to verify the reconstruction of the decay chain, further validating the `EventGenerator` implementation.  

#### Input:
None 

#### Output:
None - But drops closure mass spectra for: Truth Tops, Tops from Children, Tops from Children INIT, Truth Tops from Truth Jets and Detector leptons matched to children, and a comparison of anomalous truth to detector matching. 
___
```python
TestResonance()
```

#### Description:
In this function, a ROOT file is read into the `EventGenerator` framework and only signal tops are selected. Using their kinematic properties, the mass spectra of the BSM resonance is inferred from the tops directly and from the decay chain. This validates the implementation of the `Particle` base class, which is used extensively in the inheritance model of the ROOT objects, to provide a generic framework for particle representations. 

#### Input:
None 

#### Output:
None - But drops closure mass spectra for BSM Resonance, that is derived from: Truth Signal tops, Signal Truth Children, Signal Children (reconstructed) from Detector Leptons and Truth Jets, and purely from Detector matched truth. Non perfectly matched detector and truth objects are also compared in plots to assess the impact of improperly matched particles. 

