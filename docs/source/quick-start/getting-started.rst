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
placeholder 

.. toctree:: 
   particles
   events
