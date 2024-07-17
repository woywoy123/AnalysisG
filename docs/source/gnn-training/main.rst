Graph Neural Network Architecture
---------------------------------

**---> will write more here later <---**

Top Reconstruction using Topological Clustering 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The graph neural network being considered for this part of the documentation is known as `RecursiveGraphNeuralNetwork`, and as the name suggests, it approaches particle clustering usign a recursive approach.
In this particular case, the particle cluster should yield reconstructed top-partons, which are inferred via topological edge training.

Levels of Truth Considered for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To appropriately assess whether the Graph Neural Network is learning the features of a top-parton, various levels of truth is being included for each training step.
The first instance would be constructing top-candidates from the *Truth Children*, as the only **correct** combination of connected edges reveal the invariant mass of the top.
Based on the truth studies conducted for MC16, the invariant mass has a very narrow and distinct peak.

Following a successful outcome of the truth children reconstruction task, the truth children (excluding leptons and neutrinos) are replaced by truth jets, where the invariant mass of the top is much broader and convolved.
This is followed up with another test of removing the neutrino and using the double neutrino reconstruction algorithm, which is expected to not perfectly reconstruct the underlying truth neutrino, but rather approximate it.

The last tests will involve replacing only the truth jets with real detector jets, followed by the removal of the truth neutrino and using only detector based leptons.
A summary of is given below:

- **Truth Children**: Immediate decay Products of Tops including truth neutrinos.
- **Truth Jets**: Truth Jets including truth neutrinos and leptons.
- **Truth Jets with no Neutrino**: Truth Jets and truth leptons (except neutrino).
- **Jets**: Detector based Jets with truth neutrinos and leptons
- **Jets with no Neutrino**: Detector based Jets with truth leptons (except neutrino).
- **Detector**: Only detector based objects, i.e. Jets, Electrons and Muons.

Hyper-Parameter Tunning
^^^^^^^^^^^^^^^^^^^^^^^

Currently the hyper-parameter training schedule is as follows:

.. list-table:: Hyper-Parameter Schedule
   :header-rows: 1

   * - Name
     - Optimizer
     - Optimizer-Params

   * - MRK1
     - ADAM
     - lr: 1e-6

   * - MRK2
     - ADAM
     - lr: 1e-8

   * - MRK3
     - ADAM
     - lr: 1e-6, ams_grad : True

   * - MRK4
     - SGD
     - lr: 1e-6

   * - MRK5
     - SGD
     - lr: 1e-8

   * - MRK6
     - SGD
     - lr: 1e-5, momentum : 0.1

   * - MRK7
     - SGD
     - lr: 1e-5, momentum : 0.05, dampening : 0.01

Performance of Each Step
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree:: 
   :titlesonly:

   truthjet-truth-neutrino/main.rst
