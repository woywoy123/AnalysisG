# AnalysisG Core CXX Modules (Advanced)
## pyc (Python CUDA)
- A detached Python package that integrates high-performance native C++ and CUDA code/kernels using the PyTorch API.
- Designed for deep learning tasks, optimized for GPU performance with seamless integration between Python and CUDA kernels.
## analysis
- Implements action chains based on a user-defined template class to process events or data flow.
- Utilizes templates to define workflows that automate complex data handling processes efficiently.
## container
- A temporary storage mechanism for pointers related to graph, event, particle, and selection data types.
- Manages these pointers efficiently, ensuring optimal memory usage and performance in data processing tasks.
## dataloader
- A base class designed to coordinate `graph_t` types and optimize the serving of data to Graph Neural Networks (GNNs).
- Focuses on enhancing data loading efficiency and optimization for GNN inference and training operations.
## event
- A template class that specifies the type of event definitions and particle configurations.
- Allows users to define custom event processing logic, influencing how data is interpreted and handled within the framework.
## graph
- A template class that defines inclusive graph features such as edges, nodes, and global attributes.
- Supports various graph types with customizable attributes for flexible graph-based analysis.
## io
- A Cython interface for interacting with the ROOT C++ package, aiming for simplicity and minimal syntax.
- Facilitates handling of ROOT n-tuples, making ROOT data accessible through Python with ease.
## lossfx
- Interfaces with LibTorch's loss functions and configures them for integration into the framework.
- Provides a bridge between PyTorch's loss functions and AnalysisG's workflow, allowing custom loss configurations.
## meta
- A base class that stores metadata from ROOT files, including data fetched via PyAMI.
- Manages metadata crucial for analysis tasks, aiding in data interpretation and retrieval.
## metric
- Defines custom metrics that can be implemented by users to measure specific aspects of the analysis.
- Integrates seamlessly with other components, allowing dynamic addition of new metrics as needed.
## metrics
- Handles plotting of invariant mass distributions and loss metrics, providing visualization tools for analysis results.
- Utilizes `boost_histograms` and `mpl-hepp` for creating informative plots, aiding in result interpretation.
## model
- A template class that defines machine learning algorithms such as neural networks or decision trees.
- Integrates with other components to streamline the workflow from data processing to model training and inference.
## notification
- A base class designed to handle user verbosity during backend operations.
- Facilitates communication between framework components, ensuring timely notifications of events and statuses.
## optimizer
- A base class that serves as a foundation for various optimization strategies in machine learning models.
- Supports different optimization algorithms, enhancing model performance through iterative improvements.
## particle
- Works alongside `EventTemplate` to define the underlying particle types used in event processing.
- Enables precise configuration of particle definitions, affecting how particles are identified and processed.
## plotting
- A wrapper around `boost_histograms` and `mpl-hepp`, providing object-oriented syntax for plotting routines.
- Simplifies visualization tasks, allowing users to define plots with minimal code, enhancing data exploration.
## sampletracer
- Manages the lifecycle of container objects, ensuring proper memory handling and reference tracking.
- Aids in preventing memory leaks and improper object lifetimes during complex analyses.
## selection
- Defines algorithms for selecting events based on specific criteria, crucial for filtering relevant data.
- Allows dynamic configuration of selection criteria, enhancing flexibility in event processing workflows.
## structs
- Provides primitive data types for storing plain data and iterators, including enums for type safety.
- Offers basic building blocks that can be used to construct more complex data structures efficiently.
## tools
- A collection of utility functions aimed at simplifying various tasks within the framework's workflow.
- Includes helper functions for data manipulation, error handling, logging, and other common operations.
## typeofcasting
- Implements functions to transform primitive data types into tensors and vice versa.
- Facilitates seamless tensor operations in PyTorch by converting between basic data types and tensors as needed.

