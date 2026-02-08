NuSol Module (PyC)
==================

CUDA-accelerated neutrino reconstruction for batch processing.

Overview
--------

Located in ``pyc/pyc/nusol/``, this module provides GPU-accelerated neutrino 
reconstruction algorithms.

Key Features
------------

- Batch processing on GPU
- Multiple solutions per event
- Tensor interface
- High throughput (>1M events/sec on GPU)

API Reference
-------------

Batch Reconstruction
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import pyc.nusol as nusol
   
   # Prepare batch inputs (all on GPU)
   lepton_pt = torch.tensor([...]).cuda()
   lepton_eta = torch.tensor([...]).cuda()
   lepton_phi = torch.tensor([...]).cuda()
   lepton_E = torch.tensor([...]).cuda()
   met = torch.tensor([...]).cuda()
   met_phi = torch.tensor([...]).cuda()
   
   # Solve for all events at once
   solutions = nusol.solve_batch(
       lepton_pt, lepton_eta, lepton_phi, lepton_E,
       met, met_phi
   )
   
   # solutions.shape: (batch_size, num_solutions, 4)
   # where 4 = (pt, eta, phi, E)

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   # Set mass constraints
   nusol.set_w_mass(80.4)  # GeV
   nusol.set_top_mass(173.0)  # GeV
   
   # Set solution limits
   nusol.set_max_solutions(4)

Performance
-----------

GPU throughput:

- V100: ~1.5M events/sec
- A100: ~3M events/sec
- H100: ~5M events/sec

Compared to CPU: 100-1000x faster

See Also
--------

* :doc:`../modules/nusol` - CPU NuSol implementation
* :doc:`physics` - Physics calculations
