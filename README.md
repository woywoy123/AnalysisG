# A Graph Neural Network Framework for High Energy Particle Physics

## Introduction <a name="introduction"></a>
The aim of this package is to provide Particle Physicists with an intuitive interface to **Graph Neural Networks**, whilst remaining Analysis agnostic. 
Following a similar spirit to AnalysisTop, the analyst is able to define a custom event class and define how variables within the ROOT file are related. 
For instance, if any truth matching to specific detector particles needs to be performed, this would be defined within the event class.

The philosophy of this package is that the events within ROOT files are compiled into pythonic objects, where trees, branches and leaves define the relevant objects. 
Particle objects are defined by customized classes, which inherit some base class and relevant leaves and branches are provided as needed (see tutorial later on). 
Within the event class any additional truth matching of particles can also be done, hence the customization of events.

The second phase of the framework is to bridge the Deep Learning framework (PyTorch Geometric) and events within ROOT files. 
Similar to the event object definition, event graphs are arbitrarily defined and compiled into the PyTorch Geometric (PyG) Data object, these can be subsequently interfaced with a GraphLoader (more on this later).
Event graphs should only contain particles relevant for the analysis, for instance, if some arbitrary model is to be evaluated on Monte Carlo truth particles (simulated particle interactions/decays), only those particles should be selected in the event graph rather than detector observables. 
Once these graphs have been defined, functions can be applied to the graph, which extract relevant particle/event attributes, e.g. particle phi, missing ET, etc. and add these as features of the graph.

The final step of using the framework involves an optimization step, where models are trained to event graphs to optimize the model's internal parameters.
For larger projects additional tools such as scheduling and condor submission DAG scripts are also useful. 

## Supported Monte Carlo Generators <a name="Generators"></a>
ROOT files which originate from AnalysisTop can be easily processed and require minimal adjustment to the event compiler.

## Getting Started <a name="GettingStarted"></a>
1. First clone this repository:
```
git clone https://github.com/woywoy123/FourTopsAnalysis.git
```
2. Use the shell script to install required packages
- ```bash SetupAnalysis.sh```

3. Run the following command:
```bash 
python setup.py install 
```
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
- NOTE: When defining attributes in a particle class, it is crucial to match the strings of to the ROOT leaf name. 
As illustrated in the above code, the variable ```python self.pt``` expects a leaf name 'Particle.PT'. If a specific leaf is not found, the associated attribute will be removed and a warning will be issued upon compilation time. 

#### Inherited Functions and Variables:
When using the **ParticleTemplate** class, a number of useful functions and attributes are inherited. These are listed below; 
```python 
def DeltaR(P)
```
- This function expects another particle with the attributes; **eta, phi, pt, e** and calculates the delta R between two particles. 


```python 
def CalculateMass(lists = None, Name = "Mass"):
```
Calculates the invariant mass of the particle using **eta, phi, pt, e** attributes. 
Alternatively, if a list of particles is given, it will calculate the invariant mass of the list. 
By default this function creates two new attributes, *Mass_MeV* and *Mass_GeV*. 
To minimize redundant code, a list of particles can also be summed using python's in-built function ```python sum([...])``` and returns a new particle object.
However, this returns an integer if the list is empty.

#### To Override Functions:
Custom particle classes can also override template methods without any repercussion.


### How to define a Custom Event Class: <a name="CustomEventClass"></a>
#### Basic Example:
1. Open ```Event.py```and place the following line at the top; 
```python
from AnalysisTopGNN.Templates import EventTemplate 
from Particles import CustomParticle  
```
2. Similar to the Particle class, let the custom event inherit methods from **EventTemplate**. A simple example should look something like this:
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

The attribute **Objects** is a dictionary, which defines particle templates relevant for the event.
```python 
self.Objects = {
	"CustomParticleV1" : CustomParticleV1(), 
	"CustomParticleV2" : CustomParticleV2()
		}
```
The associated keyword **CustomParticleV1** or **CustomParticleV2** are arbitrary and appear as object attributes. For example, ```self.CustomParticleV1``` will contain only CustomParticleV1 objects.

#### The ```CompileEvent``` Method:
This method is used to define any particle relationships or perform pre-processing of the event.
For example in **Truth Matching**, a jet might originate from a top-quark which is presevered in the ROOT file through some variable, this variable can be retrieved and used to link the top and the jet.

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
In this framework uses a number of generator classes as intermediates to compile required samples. 
Familiarity with them isn't necessary, but useful, since it will provide more context around settings.

### EventGenerator:
This class takes as input the ```Event``` implementation and sample directory to compile pythonic event objects. 
These objects act as containers and retain file traces for each event, i.e. which ROOT files were used to compile the event.
To uniquely tag events the MD5 hash is computed and can be used to reverse look-up the original ROOT filename.
This class should be used to debug or develop preliminary analysis strategies.  

### GraphGenerator:
The ```EventGenerator``` interfaces with the ```GraphGenerator``` to convert ```Event``` objects into ```EventGraphs```, where particles are nodes, and relationships are edges.
For graphs to have any meaning, they require features.
Typical features to include are the particle's pt, eta, phi, etc., which can be easily added by using Python functions (more on this later).
Naturally, the same logic is applicable to the event graph and edges.

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


