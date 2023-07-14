Using the Framework on ROOT Analysis Samples
============================================
To make the framework compatible with your ROOT samples, first create three directories 

.. code-block:: console 

    mkdir Particles && mkdir Events && mkdir Graphs

These directories can be placed anywhere in your **Analysis** work-space, and serve to reduce clutter.
The **Particles** folder will hold particle definitions, these will be abstract representations of particle objects used to retrieve ROOT leaf content.
Similarly, the **Events** folder will hold event like abstractions, and define which particles should be retrieved from your ROOT samples. 
The **Graphs** folder is optional, but if one is interested in apply the framework to Graph Neural Networks, then this folder holds abstractions relating to generating Graph data structures.

Particle Definitions
********************
First create a new python file, for example **particles.py** and use the framework's base **ParticleTemplate** class. 
A very simple example would look like the following code. 

.. code-block:: python
    :caption: A simple Particle 
    
    from AnalysisG.Templates import ParticleTemplate 

    class CustomParticle(ParticleTemplate): # <---- Inherit functions from ParticleTemplate 

        def __init__(self):
            # Initialize the object and inherit base properties 
            ParticleTemplate.__init__(self)

            self.Type = "Particle" # <--- optional but useful for making templates 
            self.pt = self.Type + ".pt"
            self.eta = self.Type + ".eta"
            self.phi = self.Type + ".phi"
            self.e = self.Type + ".energy"

            #.... <Some other desired attributes>

An In-depth Explanation Of ParticleTemplate
*******************************************
The idea behind the **ParticleTemplate** class is that the user specifies the ROOT leaf strings for each particle, such that arbitrary names can be assigned to attributes. 
With the exception of some magic attributes (more on this later), attributes can be assigned to arbitrary string within the ROOT samples.
In the back-end, the framework will scan these attribute strings and check whether they are present within the ROOT samples. 
One might wonder why in the above example no tree or branch declaration was made, this is because this will be assigned by the **Event** implementation. 
This approach allows one to define particle classes once and recycle the code for different trees or branches. 

Event Definitions
*****************
To create an event implementation, simply navigate to the **Event** folder and create a new python file called for example, **event.py**. 
Similar to the **particle.py** case, the name of the file can be completely arbitrary. 
Within the **event.py** file, simply use the **EventTemplate** and add a few declarations, as below:

.. code-block:: python 
   :caption: A simple Event

    import sys 
    sys.path.append("../Particles")
    from particles import CustomParticle
    from AnalysisG.Templates import EventTemplate

    class CustomEvent(EventTemplate):
        def __init__(self):
            EventTemplate.__init__(self)

            self.Type = "Event"
            self.runNumber = self.Type + ".Number" # <--- Example event leaf variable

            # Specify the trees you want to use for each event.
            self.Tree = ["nominal", "..."] 

            # If there are any relevant branches add these as well.
            self.Branches = ["Particle"] 

            # Add particles/additional objects constituting the event
            self.Objects = {
                "ArbitraryParticleName" : CustomParticle()
            }

            # Event luminosity which is used for computing the 
            # integrated luminosity for a sum of events.
            self.weight = 1 # can be hard coded like this or assigned a string 

        def CompileEvent(self):
            # Particle names defined in self.Objects will appear 
            # in this code segment as self.<Some Random Name>.
            # For example; 
            for i in self.ArbitraryParticleName:
                self.ArbitraryParticleName[k].pt
            # returns a dictionary of particles in the event.

            # ... <Some Compiler Logic - Particle Matching etc.>


An In-depth Explanation Of EventTemplate
****************************************
Unlike the **ParticleTemplate**, the **EventTemplate** contains much more logic and adjustable parameters.
The most import attributes are the **Objects**, **Tree** and **Branches**, where the former is a dictionary specifying the particles to be used in the event and the latter being lists of string indicating the tree/branch to scan through. 
Content defined in the **Object** dictionary is used to generated the same number of particles in the event as there are in the ROOT leaf entry for that event. 
Furthermore, even if the leaf is a branch where the content is a nested **std::vector**, the framework will assume the sum of those particles. 
For example, if the leaf content has a length of 3, and it is a nested **std::vector** of length 4, then 12 particles will be generated. 




.. toctree:: 
   particles
   events
