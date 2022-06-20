# Graph Neural Network Framework for the Next Analysis Iteration of the Beyond Standard Model Four Tops Group

## Table of Contents 
1. [Introduction](#introduction)
2. [How Do I Make This Code Work With Samples?](#running)
	1. [How to define a Custom Particle Class](#CustomParticleClass)
	2. [How to define a Custom Event Class](#CustomEventClass)
	3. [How to define a Custom Event Graph Class](#CustomEventGraphClass)
3. [EventGenerator](#EventGenerator)


## Introduction <a name="introduction"></a>
This package attempts to be an Analysis agnostic **Graph Neural Network** framework, however with a primary focus on Four Top Quarks originating from the decay of a Beyond Standard Model heavy resonance Z'. Rather than rewriting the same generic template code for applying GNNs, the framework compiles individual events into Python based objects, this includes particles alike, and permits a simplified approach to debugging/producing Analysis plots. It is very unlikely that all of the information contained by a Python object is required when running a GNN, so the framework converts individual events into **Data** objects, which are interfaced with the **PyTorch-Geometric** **DataLoader**. These can be subsequently used to train and validate a given GNN model.


### Running This Code 
1. First clone this repository: 
```
git clone https://github.com/woywoy123/FourTopsAnalysis.git
```
2. Install the following packages: 
- To be determined.

--- 

## How Do I Make This Code Work With Samples? <a name="running"></a>
### There are three steps involved:
1. Define **Particles** in your ROOT samples as Python Classes.
2. Define the **Event** in your sample and how particles are matched together.
3. Define the **EventGraph** for your sample.

A few simple/complex examples can be found under; 
```bash 
Functions/Particles/Particles.py
Functions/Event/Implementations/(EventDelphes/Event).py 
Functions/Event/Implementations/EventGraphs.py
```

### Preliminaries: 
1. Create a single folder called **InsertYourAnalysisName**
2. Within this folder, create another three folders; **Particles**, **Event** and **EventGraph**

### How to define a Custom Particle Class: <a name="CustomParticleClass"></a>
#### Basic Example:
1. Navigate to: 
```bash 
cd InsertYourAnalysisName/Particles/
```
2. Create a new Python file called ```Particles.py``` 
3. Open this file and place the following line at the top; 
```python 
from Functions.Particles.ParticleTemplate import Particle as ParticleTemplate
```
4. Use this template as a base class and **let any Custom Particle Implementations inherit the template methods**.
5. A simple example should look something like this:
```python 
class CustomParticle(ParticleTemplate):
	def __init__(self):
		ParticleTemplate.__init__(self)

		self.Type = "Particle"
		self.PT = self.Type + ".PT"
		...

		# Very important that all variables are defined before calling this function. 
		self.__DefineParticle()
		
```
6. When defining attributes for a particle class, it is crucial to match the string to the appropriate ROOT leaf name. For instance, in the above code, the variable PT expects a leaf within a given branch called 'Particle.PT'. If a specific leaf is not found, the associated attribute will be removed. 

#### Inherited Functions and Variables:
When using the **Particle** class, a number of useful functions and attributes are inherited. These are listed below; 
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
Calculates the invariant mass of particles within the *Decay* and *Decay_init* attribute lists.

#### To Override Functions:
If any of the above functions are in conflict with your implementation, you can simply override them without any issues. 


### How to define a Custom Event Class: <a name="CustomEventClass"></a>
#### Basic Example:
1. Navigate to:
```bash 
cd InsertYourAnalysisName/Event
```
2. Create a new Python file called ```Event.py```
3. Open the file and place the following line at the top; 
```python
from Functions.Event.EventTemplate import EventTemplate
```
4. Similar to the Particle class, let any custom Event classes inherit methods from **EventTemplate**.
5. A simple example should look something like this:
```python  
class CustomEvent(EventTemplate):
	def __init__(self):
		EventTemplate.__init__(self)
		
		self.Type = "Event"
		self.runNumber = self.Type + ".Number" # <--- Example leaf variable
		self.Tree = ["nominal, ..."] # <-- Specify the trees you want to use for each event.
		self.Branches = ["Particle"] # <-- If there are any relevant branches add these as well.
		self.Objects = {
			"Particle" : Particle()
				}

		# Define all the attributes above this function.
		self.DefineObjects()
	
	def CompileEvent(self, ClearVal): 
		self.CompileParticles(ClearVal) # <-- Compile defined particles
		self.Particle = self.DictToList(self.Particle) # <-- Convert dictionary to list
```

#### The ```self.Object``` attribute:

The attribute **Objects** is a dictionary of particle implementations. For example, to include our **CustomParticle** implementation in our event, we would import this class into the ```Event.py``` and do the following:
```python 
self.Objects = {
	"Particle" : Particle(), 
	"CustomParticle" : CustomParticle()
		}
```
The associated keyword **CustomParticle** or **Particle** are arbitrary, since during the compilation these keys will appear as dictionary attributes. For example, ```self.CustomParticle``` will contain all compiled particles for each event in the ROOT sample.

#### The ```CompileEvent``` function call:
Within this function you define the relationships between particles, for instance the **Truth Matching**. The event should begin with calling the ```self.CompileParticle``` method. This ensures all particles are available as highlighted in the previous section. 
For example, the ```CustomParticles``` can be retrieved via the ```self.CustomParticles``` attribute. This will return an indexed dictionary of particles within the specific event. The ```ClearVal``` function input allows you to clean up any residual objects or attributes, as shown with the ```CompileParticles``` method call.

### How to define a Custom Event Graph Class: <a name="CustomEventGraphClass"></a>
#### Basic Example:
1. Navigate to:
```bash 
cd InsertYourAnalysisName/EventGraph
```
2. Create a new Python file called ```EventGraph.py```
3. Open this file and place the following line at the top; 
```python
from Functions.Event.EventGraphTemplate import EventGraphTemplate
```
4. Similar to the Particle class, let any custom Event Graph classes inherit methods from **EventGraphTemplate**.
5. A simple example should look something like this:
```python 
def CustomEventGraph(EventGraphTemplate):
	def __init__(self, Event):
		EventGraphTemplate.__init__(self)
		self.Event = Event 
		self.Particles += <Event.YourParticles> # <-- Basically import the particles
```


## The ```EventGenerator``` <a name="EventGenerator"></a>
The **EventGenerator** class is responsible for compiling events and reading ROOT files within a given directory. This class also tracks the individual files used for each event index, and stores the compiled events. 

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
