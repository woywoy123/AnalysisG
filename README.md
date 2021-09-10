# Analysis Package for the FOURTOPS Analysis
## Introduction 
This package is dedicated for a future iteration of the *Four Tops* analysis using *Graph Neural Networks*. The package expects ROOT files, that are derived from skimmed down TOPQ DOADs, using the **BSM4topsNtuples** package (https://gitlab.cern.ch/hqt-bsm-4tops/bsm4topsntuples). In this framework, particle objects are converted into python particle object, that are then converted into a graph data structure and subsequently converted into the **DataLoader** framework, supported by *PyTorch-Geometric*. PyTorch-Geometric is a codebase, that uses the PyTorch framework as a foundation to provide an intuitive interface to construct models, which can be applied to non euclidean data structures. 

## Folder Structure 
The codebase has the following folder structure:
- Closure : Used for testing and validating the individual functions associated with the framework.
- Functions : Container used to hold sub-functions and classes 
...- Event : Contains classes involving event generation. It reads in ROOT files and converts event objects into particles objects constituting an event 
...- GNN : Contains classes and functions relating to Graph Neural Networks. Classes include; Basic training Optimizer, GNN Model, Events to Graph, Graphs to native PyG DataLoader and Performance metrics. 
...- IO : Reads / Writes directories and does all the UpROOT reading including NUMPY conversion. 
...- Particles : Definition of particles using a generic particle object that is inherited by other particle types; Jets, Leptons, Truth, etc. 
...- Plotting : Simple plotting functions for Graphs and diagnostics histograms. 
...- Tools : Some repetitive functions that are found to be useful throughout the code. 
- Plots : Container that holds plots used for diagnostics or future histograms used for presentations 
- logs : Debugging or other interesting output. 

## Python File Index
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

## Python Functions and Classes
This section will be dedicated to give a brief description of what functions are in which files and how they are interfaced. 

###  Event.py

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



