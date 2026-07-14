Framework Architecture
======================

The AnalysisG framework is engineered to eliminate the severe I/O bottlenecks and computational overhead typically associated with High Energy Physics (HEP) Python analysis. It achieves this by moving almost the entirely of the execution—file reading, graph compilation, event selection, and deep learning tensor operations—into a highly-optimised C++ backend, while exposing a clean, dynamic Python API to the end user.

This section provides a deep-dive into the core architectural pillars of the framework.

1. Memory Management & Cython Bridge
------------------------------------

The fundamental challenge of a hybrid C++/Python framework is passing large amounts of complex data (like decay trees of physics particles) across the boundary without incurring massive serialisation penalties. 

AnalysisG solves this by strictly controlling memory ownership in C++:
- When the `Analysis` compiler runs, all particles (`particle_template`) and events (`event_template`) are dynamically heap-allocated in C++. 
- The framework uses a thin **Cython layer** solely to pass memory *pointers* between Python and C++. 
- Python does not copy or own the physics data. It merely holds a reference to the C++ objects. When an event passes out of scope, the C++ garbage collection routine safely deallocates the decay trees, preventing memory leaks and entirely bypassing Python's Global Interpreter Lock (GIL) limitations.

2. Multithreaded Execution Engine
---------------------------------

Reading millions of events from `.root` files is an inherently slow process. To maximize throughput, AnalysisG employs a highly concurrent execution engine built on native C++ threading (`std::thread` and internal thread-pool management):

- **Data Parallelism:** The framework splits ROOT tree processing across multiple worker threads. Each thread receives an isolated chunk of the dataset to prevent race conditions.
- **Concurrent Compilation:** The `CompileEvent` hook (which builds the particles) and the `selection_template` / `graph_template` transformations run concurrently across these chunks. 
- **Aggregation:** Once threads complete their workloads, the framework triggers a `merge` protocol (e.g., aggregating selection pass/fail statistics), safely joining the results back into the main process thread.

3. LibTorch Integration 
-----------------------

AnalysisG natively binds to **LibTorch** (the C++ backend of PyTorch). 

It explicitly **avoids intermediate graph wrappers like PyTorch Geometric (PyG)**. Instead, the `graph_template` generates native, flat C++ arrays for node features, edge lists, and truth labels, which are directly converted into highly efficient `torch::Tensor` objects.

- **Direct Forward Passes:** Models inheriting from `model_template` implement their `forward()` passes in C++. 
- **CUDA Acceleration:** Tensors are seamlessly moved between the CPU and GPU using LibTorch's device APIs before inference or back-propagation, bypassing Python-level overhead entirely.

4. HDF5 Data Caching
--------------------

Translating complex physics structures from ROOT trees into machine-learning ready tensors is computationally expensive. To prevent redundant computation during iterative model training, AnalysisG implements a robust caching mechanism:

- If `BuildCache = True` is set, the framework serializes the fully compiled, `torch::Tensor`-backed graphs into chunked HDF5 files on disk.
- On subsequent runs, the framework intelligently skips the `.root` file processing entirely, streaming the `torch::Tensor` batches directly from the HDF5 cache into the LibTorch neural network optimizer.

5. The NuSol Backend
--------------------

The **Neutrino Solutions (NuSol)** module is a specialised C++ backend for the kinematic reconstruction of neutrinos (particularly in top-quark decays). 

Unlike traditional methods that rely on slow, iterative numerical minimizers, NuSol implements an elegant, analytical **conformal geometric solver**. 
- It represents the kinematic constraints (e.g., $W$-boson mass, Top-quark mass) as a **pencil of conics** in a mathematical parameter space.
- The framework uses highly optimized root-finding algorithms to extract the geometric intersections of these conic objects, yielding exact analytical solutions for the neutrino four-momenta.
- These solvers are fully vectorized and integrated with CUDA via the `pyc` sub-package, allowing millions of events to be reconstructed concurrently on the GPU.
