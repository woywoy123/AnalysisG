Graph Neural Network Architecture
---------------------------------

Initialization Step
^^^^^^^^^^^^^^^^^^^

The algorithm begins with initializing a weight buffer with null entries and is used to start the recursive neural network.
Following buffer initialization, a matrix is constructed which maps the source and destination node pairs to a particular edge index.
This step is important to ensure updating the edge weights is consistent with the prior predictions.

Recursive Step
^^^^^^^^^^^^^^

Using the buffer weights, a node message is passed from the source to the destination nodes.
The message includes the node embedding features (mass, cartesian 4-vector, weight buffer), the edge invariant mass of the node state pairs (paths), the :math:`\Delta R` between nodes, and the state transition difference (:math:`n^{k+1}_i = \gamma(n^{k}_i, n^{k}_j - n^{k}_i)`).
These features are used to make a binary prediction whether the given edge should be connected or not at the current iteration.
If the edge is found to be connected, the source is added to the destination node, and the node embedding state is updated (mass, cartesian 4-vector, weight buffer).

Following an update, the weight buffers are updated using the scaling :math:`\gamma^{k+1}_{ij} = \gamma^{k}_{ij} \times softmax(\gamma^{k}_{ij},-1)` and the selected edges are removed from the considered edge collection.
The latter step allows the edge predictor to reconsider the edge given the new node state.
The motivation behind this approach is to allow the node 4-vector to represent a new pseudo-particle, which if added to a different node produces a physical particle.

An example of this would be the top-decay products, where a top decays initially into a :math:`W`-boson and :math:`b`-jet, with the :math:`W` forming either a neutrino and lepton pair, or two quark jets.
In this example, edges connecting a :math:`W` decay product with the b-jet, would produce a random distribution, whereas edges connecting decay products of the :math:`W` would produce an invariant edge mass consistent with the :math:`W` mass.
Once the :math:`W`-boson has been constructed from the binary edge predictor, the new node state can be connected to the :math:`b`-jet, producing the original top.

Top Reconstruction using Topological Clustering 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The graph neural network being considered for this part of the documentation is known as `RecursiveGraphNeuralNetwork`, and as the name suggests, it approaches particle clustering using a recursive approach.
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
