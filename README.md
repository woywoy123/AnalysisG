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
Additionally, Delphes samples originating from MadGraph5, are also readily supported and requires no additional event customization. 

## Getting Started <a name="GettingStarted"></a>
1. First clone this repository:
```
git clone https://github.com/woywoy123/FourTopsAnalysis.git
```
2. Install the following packages: 
- ```PyTorch Geometric```
- ```PyTorch``` 
- ```ONNX```
- ```UpROOT```
- ```HDF5```

3. Navigate to:
```bash 
src/PyTorchCustom/Source/
```
4. Run the following command:
```bash 
python setup.py install 
```
5. Followed by navigating back to the root directory of the repository and running the command:
```bash 
python setup.py install 
```
6. Now the package has been installed and can be used in any directory. 

---

## How Do I Make This Code Work With My Samples? <a name="CustomSamples"></a>

### There are three steps involved:
1. Define **Particles** in your ROOT samples as Python Classes, i.e. specify the trees/branches/leaves which define your particle.
2. Define the **Event**. This is where any particle matching is defined, for instance detector observables originating from a top-quark can be matched and stored as "truth" information.
3. Define the **EventGraph**. This is the most trivial step, simply select the particles relevant for the analysis and the compiler will do the rest.

A few simple/complex examples can be found under; 
```bash 
src/EventTemplates/Particles/Particles.py
src/EventTemplates/Event.py
src/EventTemplates/EventGraphs.py
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
		self.PT = self.Type + ".PT"
		...
		
		self.Decay = []
		self.Decay_init = []
		# Very important that all variables are defined before calling this function. 
		self._DefineParticle() 
		
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

```python 
def CalculateMassFromChildren():
```
Calculates the invariant mass of particles within the *Decay* and *Decay_init* attribute lists (Define these variables manually in the Particle definition as shown above).

#### To Override Functions:
If any of the above functions are in conflict with your implementation, you can simply override them without any issues. 


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

		# Define all the attributes above this function.
		self.DefineObjects()
	
	def CompileEvent(self, ClearVal): 
		self.CompileParticles(ClearVal) # <-- Compile defined particles and do garbage collection (ClearVal = True)
		self.ArbitraryParticleName = self.DictToList(self.ArbitraryParticleName) # <-- Convert dictionary to list
```

#### The ```self.Object``` attribute:

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
		self.Particles += <Event.YourParticles> # <-- Select particles relevant for the analysis.
```
## Generator Classes: 
In this framework generator classes are used to either compile or interface the analysis samples with the Deep Learning realm. 
A short summary is given below of each.

### EventGenerator:
This class takes as input some ```Event``` implementation and sample directory, and compiles ROOT files into Python based objects. 
These objects act as containers and retain file traces for each event.
Generally, this class should be used for debugging or developing preliminary analysis strategies, since all specified ROOT leaves are available for each Particle/Event. 

### GenerateDataLoader:
This class processes the ```EventGenerator``` container and transforms ```Event``` objects into event graphs, where particles are nodes, and any relationships are represented as edges.
Prior to compilation, simple python functions can be added to extract only relevant attributes from the ```Event```/```Particle``` objects.
Generally, the output graphs are significantly smaller and are used as input for PyTorch Geometric. 
Furthermore, this class retains traces of the originating ROOT file and records the associated index of the event. 

### Optimizer:
A class dedicated solely towards interfacing with the Deep Learning realm.
```GenerateDataLoader``` containers are imported, along with some model to be tested. 
Initially, the framework will assess the compatibility between the model and sample by checking common attributes. 
Following a successful assessment, the ```Optimizer``` will begin training the model and record associated statistics (training/validation loss and accuracy). 
Once the training has concluded, additional sample information is dumped as .pkl files.

### Analysis:
This class has all adjustable parameters of the previously discussed generators and serves as a single aggregated version of the generators. 
For larger scale projects, it is highly recommended to use this class, since it invokes all of the above classes in a completely configurable way. 
Additionally, this class can interface with the ```Submission``` module, which contains a Condor Directed Acyclic Graph compiler. 







----

############### Old Stuff.... Ignore ##########


### Importing and Basic Usage:
1. First import the class to your workspace along with the appropriate **Event** implementation; 
```python 
from Functions.Event.EventGenerator import EventGenerator
from InsertYourAnalysisName.Event.Event import <YourEventName>
```
2. Initialize the class and assign the ```EventImplementation``` to your **```<YourEventName>```** implementation. 
3. Call ```SpawnEvents()```
4. Call ```CompileEvent()```

### Inputs:
```python 
EventGenerator(dir = None, Verbose = True, Start = 0, Stop = -1)
```
- ```dir```: Expects a string of the directory/sample where the ROOT samples are located. 
- ```Verbose```: Changes the verbosity of the output, with ```False``` being none.
- ```Start```: At which event the compiler should start. This value holds globally and not per ROOT file.
- ```Stop```: At which event the compiler should terminate. 

### Attributes:
- ```Events```: A dictionary of dictionary, where the first dictionary indicates the event number, and the second the tree used for each event.
- ```FileEventIndex```: A dictionary used for book-keeping event to ROOT file association.
- ```Files```: Files contained in given directory. 
- ```Threads```: Number of CPU threads available for compiling the events.
- ```VerboseLevel```: Degree of verbosity during compilation.
- ```EventImplementation```: **Used for compiling the events in the given ROOT samples**

### Methods:
```python 
def SpawnEvents()
```
Scans the given ROOT files for required Trees/Branches/Leaves and alerts if any are missing. 

```python 
def CompileEvent(SingleTread = False, ClearVal = True)
```
This method needs to be called after the files have been scanned. The number of CPU threads can be manually set to a single thread, this is particularly useful for debugging customized event implementations.

```python 
def EventIndexFileLookup(index)
```
Returns the file used to compile the event at the given index.

## The ```GenerateDataLoader``` <a name="GenerateDataLoader"></a>
This class is used to convert events within an **EventGenerator** into graph representations, which can then be used to construct training and validation samples.

### Importing and Basic Usage:
1. Provided you already have an **EventGenerator** instance, import the following into your workspace; 
```python 
from Functions.Event.DataLoader import GenerateDataLoader
from InsertYourAnalysisName.EventGraph.EventGraph import <YourEventGraphName>
```
2. Add functions to this class through ```Add<...>Feature(name, fx)```
3. Add the **EventGenerator** instance to the generator via the ```AddSample(EventGeneratorInstance, Tree)``` function call.

### Inputs:
```python 
GenerateDataLoader()
```

### Attributes:
- ```Verbose```: Changes the verbosity of the output, with ```False``` being none.
- ```Device```: Defines the device where to store the graphs. Set to 'cpu' by default.
- ```NEvents```: Total number of events to process 
- ```CleanUp```: **Deletes the processed Event objects** from the EventGenerator. This is useful to reduce memory usage.
- ```DataContainer```: A dictionary collection of all processed EventGraphs
- ```TrainingSample```: A collection of all training EventGraphs, sorted according to the number of nodes. 
- ```ValidationSample```: A collection of all validation EventGraphs, sorted according to the number of nodes. 
- ```FileTraces```: A dictionary indexing EventGraphs to File association. 

### Graph Features and Callable Functions:
In the context of Machine Learning, a feature is some numerical value(s) we provide to the algorithm, such that it can make a prediction. For instance, a particle's invariant mass would be an interesting feature for classifying the particle's origin. When working with Graph Neural Networks, this feature would be considered a **Node** feature, since particles can be abstracted as nodes on a graph and any relation between adjacent nodes would indicate these particles to be related by some property, for example their parent.

To minimize the code overhead, the class expects functions to have the appropriate number of inputs. Examples for encoding Graph, Nodes and Edge features are shown below. 

```python 
def GraphNumber(a): # <-- Graph Feature
	return a.Number

def ParticlePT(a): # <-- Node Feature
	return a.pt

def DeltaR(a, b): # <-- Edge Feature - Note there are two inputs, one for each particle involved in the edge.
	return a.DeltaR(b)
```
Now specify these functions in your GenerateDataLoader instance as shown; 
```python 
G = GenerateDataLoader()
... # <-- some other stuff
G.AddGraphFeature("Number", GraphNumber)
G.AddNodeFeature("PT", ParticlePT)
G.AddEdgeFeature("dR", DeltaR)
```

Similarly, if the GNN is an supervised model, you need to provide it some truth attributes to ensure the algorithm updates its weights according to its prediction accuracy.  

### Methods:
```python 
def AddGraphFeature(name, fx)
```
Applies an **arbitrary function, fx,**  to the EventGraph under the **name** specified.


```python 
def AddGraphTruth(name, fx)
```
Applies an **arbitrary function, fx,**  to the EventGraph under the **T_ + name** specified.


```python 
def AddNodeFeature(name, fx)
```
Applies an **arbitrary function, fx,**  to each particle in the graph under the **name** specified.


```python 
def AddNodeTruth(name, fx)
```
Applies an **arbitrary function, fx,**  to each particle in the graph under the **T_ + name** specified.



```python 
def AddEdgeFeature(name, fx)
```
Applies an **arbitrary function, fx,**  to each particle pair in the graph under the **name** specified.


```python 
def AddEdgeTruth(name, fx)
```
Applies an **arbitrary function, fx,**  to each particle pair in the graph under the **T_ + name** specified.


```python
def SetDevice(device, SampleList = None)
```
Changes the **device** storing EventGraphs contained in the **DataContainer** if no **SampleList** has been provided.


```python 
def AddSample(EventGraphInstance, Tree, SelfLoop = False, FullyConnect = True)
```
- ```EventGeneratorInstance```: Insert an EventGenerator instance. 
- ```Tree```: Specifies the tree to use when converting the **Event** object from the EventGenerator.
- ```SelfLoop```: Add edges where the source and receiver node are the same. 
- ```FullyConnect```: Connect all particles in the event with edges. 

```python 
def MakeTrainingSample(ValidationSize = 50)
```
Populates the **Training(Validation)Sample** dictionaries from the **DataContainer** according to the percentage specified by **ValidationSize**. 
