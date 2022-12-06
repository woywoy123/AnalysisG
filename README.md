# A Graph Neural Network Framework for High Energy Particle Physics

## Introduction <a name="introduction"></a>
The aim of this package is to provide Particle Physicists with an intuitive interface to **Graph Neural Networks**, whilst remaining Analysis agnostic. 
Following a similar spirit to AnalysisTop, the analyst is able to define a custom event class and define how variables within the ROOT files are related. 
For instance, if any truth particle matching to specific detector particles is needed, this would be defined within the event class.

The philosophy of this package is that the events within ROOT files are compiled into pythonic objects, where trees, branches and leaves define the relevant object's attributes (e.g. PT, eta, phi, energy,.,..). 
To create custom particle objects, a base class is inherited, which interprets and compiles the particle as needed (see tutorial below).which interprets and compiles the particle as needed (see tutorial below).
These particle objects live within event objects that can be used to introduce further complexity, such as truth matching and so forth. 

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

# Analysis Class:
To start using this class, simply place ```python from AnalysisTopGNN import Analysis``` at the beginning of the analysis project.
The class itself is a simple wrapper for ```EventGenerator```, ```GraphGenerator``` and ```Optimization``` classes, and can be used to completely initialize the analysis without explicitly importing these classes.

## Getting Started With Your Analysis - Simple Example
A very simple analysis could look something like this;
```python 
from AnalysisTopGNN import Analysis 
from SomeEventImplementation import CustomEvent

Ana = Analysis()
Ana.ProjectName = "Example"
Ana.InputSample(<name of sample>, "/some/sample/directory") # This will scan the entire 'directory' folder for .root files
Ana.Event = CustomEvent # Import the event implementation as shown in the example above
Ana.EventCache = True # Tells the compiler to use local
Ana.Launch()

for event in Ana:
	print(event)
``` 
The above code sniplet will effectively scan for .root files within the given directory and compile associated entries into python objects within the "Example/<name of sample>" directory. 
By default, the analysis wrapper will use multithreading (12 Cores) to increase compilation speed. 
However, the above code lacks the ability to store compiled events, making recompiling time consuming and inefficient. 
Adding the line ```python Ana.DumpPickle = True``` before the ```python Ana.Launch()``` command will dump the compiled events as .pkl files.

### The EventContainer 
When iterating over the Analysis wrapper (as shown in the above code), EventContainer objects are returned. 
These contain a number of attributes which track the origin of this event, including the trees from which the customized events are compiled.



# Object Classes, Attributes and Functions:
This section lists attributes for classes.

## Generators.Analysis:
- VerboseLevel (int): The level of verbosity of the object, where 1 being the lowest and 3 the highest.
- chnk (int): Number of entries that multithreading will compile per job. This option is used to avoid excessive memory usage. 
- Threads (int): Number of simultaneous threads to use (default is 12)
- Tree (str): If the ROOT file contains multiple trees, this specifies the tree to compile and ignores others.
- EventStart (int): The event index that compilation starts from.
- EventStop (int): The event index where compilation should terminate for each ROOT file.
- ProjectName (str): The name of the Analysis. This will generate a folder by the given name (default is 'UNTITLED')
- OutputDirectory (str): The directory where the Analysis should be saved.
- Event (Event Class): The event implementation to be used for the compilation
- EventGraph (Event Graph Class): Defines which particles should be used to construct a graph.
- FullyConnect (bool): Constructs a fully connected graph from the EventGraph (default True)
- SelfLoop (bool): Whether to add edges connecting the node itself (i = j) (default True)
- TrainingSampleName (str): Name of the training and test sample.
- TrainingPercentage (int): Percentage of the total sample to be used for training the Graph Neural Network. The remaining will be withheld for testing the performance on unseen data. (default 80)
- SplitSampleByNode (bool): Whether to partition the training sample into n-Node training samples (useful for GNNs requiring identical n-Node samples). (default False)
- kFolds (int): Number folds performed in the training (default 10).
- BatchSize (int): Number of event graphs to aggregate into a single larger graph. (default 10)
- Model: Model object to be trained. 
- DebugMode (str): Providing a string with keywords 'loss', 'accuracy' and 'compare' will increase the verbosity of the training and are useful for debugging.
- ContinueTraining (bool): Whether to consider previous epochs of the model training. If set to False, previous epochs will be overwritten. (default True)
- RunName (str): Name of the training project (default 'Untitled')
- Epochs (int): Number of times to iterate over the entire training sample. (default 10)
- Device (str): Using 'cuda' will trigger the code to train the model on GPU. 
- VerbosityIncrement (int): By how many percentage points to nofity the training progress (VerbosityLevel needs to be 3) (default 10)

## Plotting.TH1F:
- Title (str): Title of the histogram 
- xTitle (str): x-Axis title
- xMin (float): Starting range of the histogram 
- xMax (float): Terminating point of the histogram's range.
- xBins (int): Number of bins to use in the histogram.
- xStep (float): By how many units the axis ticks are increased by (If the xMin is provided, TH1F will automatically deduce the number of bins required to define the histogram).
- OutputDirectory (str): Output directory of the saved figure.
- Style (str): The keywords 'ATLAS', 'ROOT' are used to indicate how to present the histogram (This is based on the mplhep package).
- ATLASLumi (float): The luminosity provided to the histogram.
- NEvents (int): Number of events used to construct the histogram (default is the length of xData)
- xData (list): A list of the data to populate the histogram with.
- Filename (str): Name of the saved figure (automatically concatinates .png)
- xWeights (list): Weights being applied per bin. To use this, one needs to populate xData with a list e.g. ```python [i for i in range(len(xWeights))]```
- xTickLabels (list): override the default matplotlib histogram ticks for each bin 
- xBinCentering (bool): Whether to center the bins of a histogram (useful for illustrating classification plots).
- Logarithmic (bool): Whether to scale the y-axis logarithmically.

