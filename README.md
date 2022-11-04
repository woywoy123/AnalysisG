# A Graph Neural Network Framework for High Energy Particle Physics

## Introduction <a name="introduction"></a>
The aim of this package is to provide Particle Physicists with an intuitive interface to **Graph Neural Networks**, whilst remaining Analysis agnostic. 
Specifically, the package was partly inspired by AnalysisTop, which allows for not only event selection customization, but at the individual particle level. 
This package adopts a similar approach, processed ROOT files (MadGraph5 Delphes, AnalysisTop, ...) containing trees, branches, and leaves are read by the UpROOT package and compiled into Particle pythonic objects.
The definition of a particle object is completely arbitrary, as long as the relevant leaves and branches are provided (see tutorial later on). 
In a similar spirit, collision events are compiled into Event python objects, which also allows for flexibility in terms of how particles are related (e.g. Truth Matching), providing a simplistic first step into inspecting samples for a given analysis.

The second phase of the framework is the bridge between the Deep Learning realm and collision events. 
Similar to the event object definition, event graphs are arbitrarily defined and compiled into the PyTorch Geometric (PyG) Data object, which can be subsequently used in the DataLoader (more on this later).
Event graphs should only contain particles relevant for the analysis, for instance, if some arbitrary model is to be evaluated on Monte Carlo truth particles (simulated particle interactions/decays), only those particles should be selected in the event graph rather than detector observables. 
Once these graphs are defined, functions can be applied to the graph, which extract relevant particle/event attributes, e.g. particle phi, missing ET, etc.

The final step of using the framework would involve the usage of an optimizer, which handles all the Deep Learning (PyTorch) function calls and matches input graph attributes with defined models. 
As for larger projects, additional tools such as scheduling and condor submission DAG scripts can also be compiled. 

## Supported Monte Carlo Generators <a name="Generators"></a>
ROOT files which originate from AnalysisTop can be easily processed and require minimal adjustment to the event compiler.

## Getting Started <a name="GettingStarted"></a>
1. First clone this repository:
```
git clone https://github.com/woywoy123/FourTopsAnalysis.git
```
2. Install the following packages: 
- ```PyTorch Geometric```
- ```PyTorch``` 
- ```UpROOT```
- ```Sklearn```
- ```HDF5```

3. Run the following command:
```bash 
python setup.py install 
```
4. Now the package has been installed and can be used in any directory. 
---

## How Do I Make This Code Work With My Samples? <a name="CustomSamples"></a>

### There are three steps involved:
1. Define **Particles** in your ROOT samples as Python Classes, i.e. specify the trees/branches/leaves which define your particle.
2. Define the **Event**. This is where any particle matching is defined, for instance detector observables originating from a top-quark can be matched and stored as "truth" information.
3. Define the **EventGraph**. This is the most trivial step, simply select the particles relevant for the analysis and the compiler will do the rest.

A few simple/complex examples can be found under; 
```bash 
src/EventTemplates/Particles/Particles.py
src/EventTemplates/Events/Event.py
src/EventTemplates/Events/EventGraphs.py
```

### A Simple Example:
1. Generate a new analysis directory **InsertAnalysisName**
2. Create three files; **Particles.py**, **Event.py**, **EventGraph.py**

### Defining A Particle Class: <a name="CustomParticleClass"></a>
1. Open ```Particles.py``` and place the following line at the top; 
```python 
from AnalysisTopGNN.Templates import ParticleTemplate
```
2. Use this template as a base class and **let any Custom Particle Implementations inherit the template methods**. A simple example should look something like this:
```python 
class CustomParticle(ParticleTemplate):
	def __init__(self):
		ParticleTemplate.__init__(self)

		self.Type = "Particle"
		self.pt = self.Type + ".PT"
		... <Some Desired Particle Properties> ...
		self.eta = self.Type + ".ETA"
		
```
NOTE: When defining attributes for a particle class, it is crucial to match the string to the appropriate ROOT leaf name. For instance, in the above code, the variable PT expects a leaf within a given branch called 'Particle.PT'. If a specific leaf is not found, the associated attribute will be removed and a warning will be issued upon compilation time. 

#### Inherited Functions and Variables:
When using the **ParticleTemplate** class, a number of useful functions and attributes are inherited. These are listed below; 
```python 
def DeltaR(P)
```
This function expects another particle with the attributes; **eta, phi, pt, e** and calculates the delta R between the current particle and the given one. 

```python 
def CalculateMass(lists = None, Name = "Mass"):
```
Calculates the invariant mass of the particle's **eta, phi, pt, e** attributes, if no lists have been provided. By default this function call spawns two new attributes, *Mass_MeV* and *Mass_GeV*. If a list of particles is provided, the **total invariant mass** is calculated. 
Alternatively, given a list of particles, the ```python sum([...])``` in-built function can be used to calculate the invariant mass of a new instantiated particle object.

#### To Override Functions:
Custom particles with conflicting method names can be savely used to override pre-existing methods. 


### How to define a Custom Event Class: <a name="CustomEventClass"></a>
#### Basic Example:
1. Open ```Event.py```and place the following line at the top; 
```python
from AnalysisTopGNN.Templates import EventTemplate 
from Particles import CustomParticle  
```
2. Similar to the Particle class, let any custom Event inherit the methods from **EventTemplate**. A simple example should look something like this:
```python  
class CustomEvent(EventTemplate):
	def __init__(self):
		EventTemplate.__init__(self)
		
		self.Type = "Event"
		self.runNumber = self.Type + ".Number" # <--- Example event leaf variable
		self.Tree = ["nominal, ..."] # <-- Specify the trees you want to use for each event.
		self.Branches = ["Particle"] # <-- If there are any relevant branches add these as well.
		self.Objects = {
			"ArbitraryParticleName" : CustomParticle()
				}
		
		self.Lumi = 0 # <--- event weight used to calculate the integrated luminosity of events.
		# Define all the attributes above this function.
		self.DefineObjects()
	
	def CompileEvent(self): 
		# Particle names defined in self.Objects will appear in this code segment as self.<Some Random Name>. For example below; 
		print(self.ArbitraryParticleName) # <--- returns a dictionary of particles in the event.
		self.ArbitraryParticleName = self.DictToList(self.ArbitraryParticleName) # <-- Convert dictionary to list
		
		... <Some Compiler Logic - Particle Matching etc.>

```

#### The ```self.Objects``` attribute:

The attribute **Objects** is a dictionary of particle implementations.
```python 
self.Objects = {
	"CustomParticleV1" : CustomParticleV1(), 
	"CustomParticleV2" : CustomParticleV2()
		}
```
The associated keyword **CustomParticleV1** or **CustomParticleV2** are arbitrary and appear as object attributes. For example, ```self.CustomParticleV1``` will contain CustomParticleV1 objects for the given collision event.

#### The ```CompileEvent``` function call:
This function call is similar ```AnalysisTop's``` CustomEventSaver, here any particle relationships are defined. For instance in **Truth Matching**, a jet might originate from the decay of a top-quark, but the ROOT files may not have an explicit variable to indicate this.
Such missing variables can be applied to the particle objects and can subsequently be used for some Deep Learning Model or preliminary kinematic studies.
The event should begin with calling the ```self.CompileParticle``` method and ensures all defined particles are be available. 

### How to define a Custom Event Graph Class: <a name="CustomEventGraphClass"></a>
#### Basic Example:
1. Open ```EventGraphs.py``` and place the following line at the top; 
```python
from AnalysisTopGNN.Templates import EventGraphTemplate 
```
2. Similar to the Particle class, let any custom Event Graph classes inherit methods from **EventGraphTemplate**.
3. A simple example should look something like this:
```python 
def CustomEventGraph(EventGraphTemplate):
	def __init__(self, Event):
		EventGraphTemplate.__init__(self)
		self.Event = Event 
		self.Particles += <Event.SomeParticles> # <-- Select particles relevant for the analysis. 
```

## Generator Classes: 
In this framework generator classes are used to either compile or interface the analysis samples with the Deep Learning realm. 

### EventGenerator:
This class takes as input some ```Event``` implementation and sample directory, and compiles ROOT files into Python based objects. 
These objects act as containers and retain file traces for each event.
Generally, this class should be used for debugging or developing preliminary analysis strategies, since all specified ROOT leaves are available for each Particle/Event. 

### GraphGenerator:
This class processes the ```EventGenerator``` container and transforms ```Event``` objects into event graphs, where particles are nodes, and any relationships are represented as edges.
Prior to compilation, simple python functions can be added to extract only relevant attributes from the ```Event```/```Particle``` objects.
Generally, the output graphs are significantly smaller and are used as input for PyTorch Geometric. 
Furthermore, this class retains traces of the originating ROOT file and records the associated index of the event. 

### Optimization:
A class dedicated solely towards interfacing with the Deep Learning frameworks (specifically ```PyTorch```).
```GenerateDataLoader``` containers are imported, along with some model to be tested. 
Initially, the framework will assess the compatibility between the model and sample by checking common attributes. 
Following a successful assessment, the ```Optimizer``` will begin training the model and record associated statistics (training/validation loss and accuracy). 
Once the training has concluded, additional sample information is dumped as .pkl files.

### Analysis:
This class has all adjustable parameters of the previously discussed generators and serves as a single aggregated version of the generators. 
For larger scale projects, it is highly recommended to use this class, since it invokes all of the above classes in a completely configurable way. 
Additionally, this class can interface with the ```Submission``` module, which contains a Condor Directed Acyclic Graph compiler. 


