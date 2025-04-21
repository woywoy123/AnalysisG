As the field of High Energy Particle Physics (HEPP) has begun exploring more exotic machine learning algorithms, such as Graph Neural Networks (GNNs).
Analyses commonly rely on pre-existing data science frameworks, including PyTorch, TensorFlow and Keras, to recast ROOT samples into some respective data structure.
This often results in tedious and computationally expensive co-routines to be written.
Community projects, such as UpROOT, Awkward, and Scikit-HEP are developing tools to address some of these problems.
For instance, in the context of Graph Neural Networks, converting non-monolithic data into graphs with edge, node and graph level feature becomes increasingly complex.

AnalysisG aims to address these residual issues, by following a similar philosophy as *AnalysisTop*, whereby events and particles are treated as polymorphic objects.
The framework initially translate ROOT based n-tuples into user defined particle and event objects, where strings from ROOT samples (trees/branches/leaves) are mapped to their respective attributes.
Particles living within the event definition can be retrospectively matched with other particles to build complex decay chains, and subsequently used for truth matching studies and machine learning algorithms.

For Graph Neural Networks in particular, graph structures can be constructed via a template graph class, which will populate (graph, node and edge) feature tensors using the event and particle definitions using simple callable functions.
The resulting graphs can then be used for inference or supervised training sessions.

For preliminary cut based analyses, the framework offers selection templates, which take the prior event definitions as input and allows for detailed studies, which can then be exported into ROOT n-tuples.
Alternatively, the resulting selection templates can be assigned relevant attributes, which can be subsequently serialized and plotted as would be the case in dedicated truth studies.

To facilitate faster machine learning related training/inference in high energy particle physics, the framework exploits algorithms written entirely in C++ and native CUDA kernels.
These algorithms can be found within a self contained sub-package referred to as *pyc* and can be used outside of this framework. 
Some of the algorithms include; $\Delta R$, Polar to Cartesian coordinate system transformations, Invariant Mass computations, edge/node single counting aggregation, analytical single/double neutrino reconstruction and many more.

Given the growing trend in machine learning, across multiple collaborations, the framework aims to remain Analysis agnostic, such that mutually exclusive ATLAS/Belle-II analyses can benefit from this framework. 


