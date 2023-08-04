# A Graph Neural Network Framework for High Energy Particle Physics
[![AnalysisG-Building-Action](https://github.com/woywoy123/AnalysisTopGNN/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/woywoy123/AnalysisTopGNN/actions/workflows/test.yml)
[![AnalysisG-Coverage-Action](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/woywoy123/6fee1eff8f987ac756a20133618659a1/raw/covbadge.json)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/analysisg/badge/?version=latest)](https://analysisg.readthedocs.io/en/latest/?badge=latest)

## Introduction <a name="introduction"></a>
The aim of this package is to provide Particle Physicists with an intuitive interface to **Graph Neural Networks**, whilst remaining Analysis agnostic. 
Following a similar spirit to AnalysisTop, the physicist defines a custom event class and matches variables within ROOT files to objects that they are a representation of.
A simple example of this would be a particle, since these generally have some defining properties such as the four vector, mass, type, etc. 

From a technical point of view, the particle would be represented by some Python object, where attributes are matched to the ROOT leaf strings, such that framework can identify how variables are matched. 
A similar approach can be taken to construct event objects, where particle objects live within the event and are matched accordingly to any other particles e.g. particle truth matching. 
This hierarchical architecture allows for complex event definitions, first basic building blocks are defined and then matched according to some rule (see tutorial below).

To streamline the transition between ROOT and PyTorch Geometric (a Deep Learning framework for Graph Neural Networks), the framework utilizes event graph definitions.
These simply define which particles should be used to construct nodes on a PyTorch Geometric (PyG) Data object. Edge, Node and Graph features can be added separately as simple python functions (see tutorial below).
Post event graph construction, events are delegated to an optimization step, which trains a specified model with those graphs. 

To avoid having to deal with additional boiler plate book keeping code, the framework tracks the event to the originating ROOT file using a hashing algorithm. 
The hash is constructed by concatenating the directory, ROOT filename and event number into a single string and computing the associated hash. 
This ensures each event can be easily traced back to its original ROOT file. 


### (Optional) Define a Custom Selection Class: 
#### Basic Example:
1. Open/Create `Selection.py` and place the following line at the top; 
```python 
from AnalysisG.Templates import SelectionTemplate
```
2. Let the your selection class inherit methods from `SelectionTemplate`
3. A simple example should look something like this: 
```python 
def SomeCoolSelection(SelectionTemplate):
    def __init__(self):
        SelectionTemplate.__init__(self)
        
        # Add some attributes you want to capture in this selection 
        # This can be a nested list/dictionary or a mixture of both
        self.SomeParticleStuff = {"lep" : [], "had" : []} 
        self.SomeCounter = {"lep" : 0, "had" : 0}

    def Selection(self, event):
        if len(event.<SomeParticles>) == 0: return False # Reject the event 
        return True # Accept this event and continue to the Strategy function.

    def Strategy(self, event):
        # Recall the ROOT file from which this event is from 
        print(self.ROOTName)
        
        # Get the event hash (useful for debugging)
        print(self.hash)

        for i in event.<SomeParticles>:
            # <.... Do some cool Analysis ....>

            # Prematurely escape the function
            if i.accept: return "Accepted -> Particles"
            
            # Add stuff to the attributes:
            self.SomeParticleStuff["lep"].append(i.Mass)
            
            if i.is_lep: self.SomeCounter["lep"] += 1
```

#### Attributes:
- `(string) ROOTName`
Returns the current ROOT file this event belongs to.
- `(float) AverageTime`
Returns the average time required to process a bunch of events.
- `(float) StdevTime`
Returns the standard deviation of the time required to process a bunch of events.
- `(float) Luminosity` 
The total luminosity of a bunch of events passing the selection function. 
- `(int) NEvents`:
Number of events processed. 
- `(dict) CutFlow`:
Statistics involving events (not)-passing the `Selection` function.
If during the `Strategy` a string is returned with `->`, a key is created within this dictionary and a counter is automatically instantiated.
- `AllWeights`:
All collected event weights of (not)-passing events. 
- `SelWeights`:
Event weights collected which pass the `Selection` function.

#### Functions:
```python 
def Selection(event): 
```
Returns by default `True` but can be overridden to add custom selection criteria.

```python 
Strategy(event): 
```
A function which allows the analyst to extract additional information from events and implement additional complex clustering algorithms.

```python 
def Px(met, phi):
```
A function which converts polar coordinates to Cartesian x-component.

```python 
def Py(met, phi):
```
A function which converts polar coordinates to Cartesian y-component.

```python 
def MakeNu(list[px, py, pz]):
```
A function which generates a new neutrino particle object with a given set of Cartesian 3-momentum vector.

```python
def NuNu(quark1, quark2, lep1, lep2, event, mT = 172.5, mW = 80.379, mN = 0, zero = 1e-12)
```
Invokes the `DoubleNeutrino` reconstruction algorithm with the given quark and lepton pairs for this event. 
This function returns either an empty list, or a list of neutrino objects with possible solution vectors.

```python
def Nu(quark, lep, event, S = [100, 0, 0, 100], mT = 172.5, mW = 80.379, mN = 0, zero = 1e-12)
```
Invokes the `SingleNeutrino` reconstruction algorithm with the given quark and lepton pair for this event. 
This function returns either an empty list, or a list of neutrino objects with possible solution vectors.
The variable `S` is the uncertainty on the MET of the event. 

#### Magic Functions:
```python 
Ana = Analysis()

Sel = SomeCoolSelection()

# Use the Analysis class to run this on a single thread
Sel(Ana) 

# Adding Selections 
selected = []
for event in Ana:
    Sel = SomeCoolSelection()
    selected.append(Sel(event))
total = sum(selected)

# Equivalence 
Sel1 = SomeCoolSelection()
Sel2 = SomeOtherSelection()
Sel1 == Sel2 
Sel1 != Sel2
```

## The Analysis Class: 
This is the main interface of the package, it is used to configure the **Event/Graph** constructors, including **Graph Neural Network** training and many other things, which will be shown as an example.

### A Minimal Example:
To get started, create a new python file `<SomeName>.py` and open it.
At the top, add the following line: 
```python 
from AnalysisG import Analysis
from SomeEventImplementation import CustomEvent
```
Now instantiate the class and specify the analysis parameters.
A simple example could look like this; 
```python 
Ana = Analysis()
Ana.ProjectName = "Example"
Ana.InputSample(<name of sample>, "/some/sample/directory")
Ana.Event = CustomEvent
Ana.EventCache = True
Ana.Launch()

for event in Ana:
    print(event)
``` 

### Attributes and Functions:
#### Attributes
- `Verbose`: 
An integer which increases the verbosity of the framework, with 3 being the highest and 0 the lowest.
- `Threads`: 
The number of CPU threads to use for running the framework. 
- `chnk`: 
An integer which regulates the number of entries to process for each given core. 
This is particularly relevant when constructing events, as to avoid memory issues. 
As an example, if Threads is set to 2 and `chnk` is set to 10, then 10 events will be processed per core. 
- `EventStart`: 
The event to start from given a set of ROOT samples. Useful for debugging specific events.
- `EventStop`: 
The number of events to generate. 
- `ProjectName`: 
Specifies the output folder of the analysis. If the folder is non-existent, a folder will be created.
- `OutputDirectory`: 
Specifies the output directory of the analysis. This is useful if the output needs to be placed outside of the working directory.
- `EventCache`: 
Specifies whether to generate a cache after constructing `Event` objects. If this is enabled without specifying a `ProjectName`, a folder called `UNTITLED` is generated.
- `DataCache`:
specifies whether to generate a cache after constructing graph objects. If this is enabled without having an event cache, the `Event` attribute needs to be set. 
- `Event`: 
Specifies the event implementation to use for constructing the Event objects from ROOT Files.
- `EventGraph`:
Specifies the event graph implementation to use for constructing graphs.
- `SelfLoop`:
Given an event graph implementation, add edges to particle nodes which connect to themselves.
- `FullyConnect`:
Given an event graph implementation, create a fully connected graph.
- `TrainingPercentage`:
Assign some percentage to training and reserve the remaining for testing.
- `kFolds`:
Number of folds to use for training 
- `kFold`:
Explicitly use this kFold during training. This can be quite useful when doing parallel traning, since each kFold is trained completely independently. 
- `BatchSize`:
How many Event Graphs to group into a single graph.
- `Model`:
The model to be trained, more on this later.
- `DebugMode`:
Expects a boolean, if this is set to `True`, a complete print out of the training is displayed. 
- `ContinueTraining`:
Whether to continue the training from the last known checkpoint (after each epoch).
- `Optimizer`:
Expects a string of the specific optimizer to use.
Current choices are; `SGD` - Stochastic Gradient Descent and `ADAM`.
- `OptimizerParams`: 
A dictionary containing the specific input parameters for the chosen `Optimizer`.
- `Scheduler`:
Expects a string of the specific scheduler to use. 
Current choices are; `ExponentialLR` and `CyclicLR`. 
- `SchedulerParams`: 
A dictionary containing the specific input parameters for the chosen `Scheduler`.
- `Device`: 
The device used to run `PyTorch` training on. This also applies to where to store graphs during compilation.
- `TestFeature`: 
A parameter mostly concerning graph generation. It checks whether the supplied features are compatible with the `Event` python object. 
If any of the features fail, an alert is issued. 

#### Default Parameters:
| **Attribute**        | **Default Value**  | **Expected Type** |                    **Examples** |
|:---------------------|:------------------:|:-----------------:|:-------------------------------:|
| Verbose              |                  3 |             `int` |                                 | 
| Threads              |                  6 |             `int` |                                 |
| chnk                 |                100 |             `int` |                                 |
| EventStart           |                  0 |             `int` |                                 |
| EventStop            |               None |             `int` |                                 |
| ProjectName          |           UNTITLED |             `str` |                                 |
| OutputDirectory      |                 ./ |             `str` |                                 |
| Event                |               None |       EventObject |                                 |
| EventGraph           |               None |  EventGraphObject |                                 |
| SelfLoop             |               True |            `bool` |                                 |
| FullyConnect         |               True |            `bool` |                                 |
| TrainingName         |           `Sample` |             `str` |                                 |
| TrainingSize         |              False |             `int` |                                 |
| kFolds               |              False |             `int` |                                 |
| BatchSize            |                  1 |             `int` |                                 |
| Model                |               None |          GNNModel |                                 |
| DebugMode            |              False |            `bool` | `loss`, `accuracy`, `compare`   |
| EnableReconstruction |              False |            `bool` |                                 |
| ContinueTraining     |              False |            `bool` |                                 |
| RunName              |                RUN |             `str` |                                 |
| Epochs               |                 10 |             `int` |                                 |
| Optimizer            |               None |            `str`  |      `"ADAM"`                   |
| Scheduler            |               None |            `str`  |      `"CyclicLR"`               |
| OptimizerParams      |               {}   |            `dict` | `{"lr": 1e-3, "weight_decay": 1e-3}` |
| SchedulerParams      |               {}   |            `dict` | `{"base_lr": xxx, "max_lr": xxx}`    |
| Device               |                cpu |            `str`  |			            `cuda`    |
| EventCache           |              False |            `bool` |                                 |
| DataCache            |              False |            `bool` |                                 |
| TestFeatures         |              False |            `bool` |                                 |

#### Functions:
```python 
def InputSample(Name, SampleDirectory)
```
This function is used to specify the directory or sample to use for the analysis. 
The `Name` parameter expects a string, which assigns a name to `SampleDirectory` and is used for book-keeping. 
`SampleDirectory` can be either a string, pointing to a ROOT file or a nested dictionary with keys indicating the path and values being a string or list of ROOT files. 

```python 
def AddSelection(Name, inpt)
``` 
The `Name` parameter specifies the name of the selection criteria, for instance, `MyAwesomeSelection`. 
The `inpt` specifies the `Selection` implementation to use, more on this later. 

```python 
def MergeSelection(Name)
```
This function allows for post selection output to be merged into a single pickle file. 
During the execution of the `Selection` implementation, multiple threads are spawned, which individually save the output of each event selection, meaning a lot of files being written and making it less ideal for inspecting the data.
Merging combines all the internal data into one single file and deletes files being merged. 

```python 
def Launch
```
Launches the Analysis with the specified parameters.

```python 
def DumpSettings: 
``` 
Returns a directory of the settings used to configure the `Analysis` object. 

```python 
def ImportSettings(inpt):
```
Expects a dictionary of parameters used to configure the object.

```python 
def Quantize(inpt, size):
```
Expects a dictionary with lists of ROOT files, that need to be split into smaller lists (defined by size).
For instance, given a size of 2, a list of 100 ROOT files will be split into 50 lists with length 2.

#### Magic Functions:
```python 
# Iteration
[i for i in Analysis]

# Length operator
len(Analysis)

# Summation operator 
Analysis3 = Analysis1 + Analysis2
AnalysisSum = [Analysis1, Analysis2, ..., AnalysisN]
```
## AnalysisG.IO.UpROOT:
This class is predominantly designed to be interfaced with core modules in the framework. 
However, it can be used as a completely standalone module with minimal configuration. 
Part of this module is the so called `MetaData` object. 
This class contains additional information about the input ROOT samples if PyAMI is enabled. 
If PyAMI is not installed or authenticated to, then it will try to scrape the ROOT files for additional meta data. 

### Functions: 
```python 
def InputSamples(input: Union[str, Dict, List])
```
This function will scan the given input for ROOT files. 
If the input is a string containing the `.root` extension, then only that file will be used, otherwise it will assume the input is a directory and scan it for possible ROOT files.
For lists, the function will assume these to be `.root` files and never directories. 
If the input is a dictionary, then the keys can be interpreted as being directories, with values being either lists of ROOT files to read, or single ROOT file strings.

### Attributes: 
- **Trees**: Expects a list of strings pointing to the trees to be read.
- **Branches**: Expects a list of strings pointing to any branches.
- **Leaves**: Expects a list of strings pointing to any leaves to be read.
- **ScanKeys**: Will check whether the given `Trees/Branches/Leaves` are found within the ROOT samples.


## AnalysisG.Plotting.TH1F/CombineTH1F/TH2F
A class dedicated to plotting histograms using the `mplhep` package as a backend to format figures.
This class adds some additional features to simplify writing simple plotting code, such as bin centering. 

### Attributes and Functions:
#### Attributes (Cosmetics): 
- **Title**: 
Title of the histogram to generate.
- **Style**:
The style to use for plotting the histogram, options are; `ATLAS`, `ROOT` or `None`
- **ATLASData**:
A boolean switch to distinguish between *Simulation* and *Data*.
- **ATLASYear**:
The year the data/simulation was collected from.
- **ATLASCom**:
The *Center of Mass* used for the data/simulation.
- **ATLASLumi**:
The luminosity to display on the `ATLAS` formated histograms. 
- **NEvents**:
Displays the number of events used to construct the histogram. 
- **Color**:
The color to assign the histogram.
- **FontSize**:
The front size to use for text on the plot.
- **LabelSize**:
Ajusts the label sizes on the plot.
- **TitleSize**:
Modify the title font size.
- **LegendSize**:
Modify the size of the legend being displayed on the plot.
This is predominantly relevant for combining `TH1F` histograms.
- **xScaling**:
A scaling multiplier in the x-direction of the plot.
This is useful when bin labels start to merge together.
- **yScaling**:
A scaling multiplier in the y-direction of the plot.
This is useful when bin labels start to merge together.

#### Attributes (IO):
- **Filename**: 
The name given to the output `.png` file.
- **OutputDirectory**: 
The directory in which to save the figure. 
If the directory tree is non-existent, it will automatically be created.
- **DPI**:
The resolution of the figure to save. 

#### Attributes (Axis):
- **xTitle**: 
Title to place on the x-Axis.
- **yTitle**: 
Title to place on the y-Axis.
- **xMin**: 
The minimum value to start the x-Axis with.
- **xMax**:
The maximum value to end the x-Axis with.
- **yMin**: 
The minimum value to start the y-Axis with.
- **yMax**:
The maximum value to end the y-Axis with.
- **xTickLabels**:
A list of string/values to place on the x-Axis for each bin. 
The labels will be placed in the same order as given in the list.
- **Logarithmic**:
Whether to scale the bin content logarithmically.
- **Histograms**:
Expects `TH1F` objects from which to construct the combined histogram.
- **Colors**:
Expects a list of string indicating the color each histogram should be assigned.
The `CombineTH1F` automatically adjusts the color if a color has been assigned to another histogram.
- **Alpha**:
The alpha by which the color should be scaled by. 
- **FillHist**:
Whether to fill the histograms with the color assigned or not.
- **Texture**:
The filling pattern of the histogram, options are; `/ , \\ , | , - , + , x, o, O, ., *, True, False`
- **Stack**:
Whether to combine the histograms as a stack plot.
- **Histogram**:
A single `TH1F` object to which other `Histograms` are plotted against. 
- **LaTeX**:
Whether to use the *LaTeX* engine of `MatplotLib`

#### Attributes (Bins):
- **xBins**:
The number of bins to construct the histogram with.
- **xBinCentering**:
Whether to center the bins of the histograms. 
This can be relevant for classification plots.
- **xStep**:
The step size of placing a label on the x-Axis, e.g. 0, 100, 200, ..., (n-1)x100.
- **yStep**:
The step size of placing a label on the y-Axis, e.g. 0, 100, 200, ..., (n-1)x100.

#### Attributes (Data):
- **xData**:
The data from which to construct the histogram. 
If this is to be used with `xTickLabels`, make sure the bin numbers are mapped to the input list.
For example; `xData = [0, 1, 2, 3, 4]  -> xTickLabels = ["b1", "b2", "b3", "b4", "b5"]`
- **xWeights**:
Weights to be used to scale the bin content. 
This is particularly useful for using `xTickLabels`.
- **Normalize**:
Whether to normalize the data. Options are; `%`, `True` or `False`.
- **IncludeOverflow**:
Whether to dedicate the last bin in the histogram for values beyond the specified maximum range.

#### Functions (IO):
```python 
def DumpDict(varname = None)
``` 
Dumps a dictionary representation of the settings.
```python 
def Precompiler()
``` 
A function which can be overridden and is used to perform preliminary data manipulation or histogram modifications.
```python 
def SaveFigure(Dir = None)
```
Whether to compile the given histogram object. 
`Dir` is a variable used to indicate the output directory. 

#### Functions (Cosmetics): 
```python 
def ApplyRandomColor(obj)
```
Selects a random color for the histograms.

```python 
def ApplyRandomTexture(obj)
```
Selects a random texture for the histograms.

## Submission.Condor:
- AddJob 
- LocalDryRun 
- DumpCondorJobs 

## IO:
- Pickle.PickleObject
- Pickle.UnpickleObject
- Pickle.MultiThreadedDump
- Pickle.MultiThreadedReading

