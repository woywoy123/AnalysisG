NuSol Module
============

The NuSol (Neutrino Solution) module provides algorithms for neutrino reconstruction in particle physics.

Overview
--------

Located in ``src/AnalysisG/modules/nusol/``, this module implements various neutrino reconstruction algorithms:

- **NuSol**: Standard neutrino reconstruction
- **Ellipse**: Ellipse-based methods
- **Conuix**: Conic section intersection methods
- **MultiSol**: Multiple solution handling

These algorithms solve the challenging problem of reconstructing neutrino 4-momenta from incomplete 
kinematic information in particle physics events.

Neutrino Reconstruction Problem
--------------------------------

In particle physics, neutrinos are not detected directly, but their presence is inferred from:

- Missing transverse energy (MET)
- Kinematic constraints (e.g., W boson mass)
- Top quark decay topology

The reconstruction typically involves solving a system of nonlinear equations with:

- Known: Lepton kinematics, MET, jet kinematics
- Unknown: Neutrino longitudinal momentum (pz) and potentially complete 3-momentum

Algorithms
----------

NuSol Algorithm
~~~~~~~~~~~~~~~

Standard neutrino reconstruction using W-boson mass constraint:

.. code-block:: python

   from AnalysisG.modules.nusol import NuSol
   
   # Create solver
   solver = NuSol()
   
   # Set inputs
   solver.lepton_pt = lepton.pt
   solver.lepton_eta = lepton.eta
   solver.lepton_phi = lepton.phi
   solver.lepton_E = lepton.E
   
   solver.met = event.met
   solver.met_phi = event.met_phi
   
   # Solve
   solutions = solver.Solve()
   
   # Access solutions
   for sol in solutions:
       nu_pt = sol.pt
       nu_eta = sol.eta
       nu_phi = sol.phi
       nu_E = sol.E

Ellipse Method
~~~~~~~~~~~~~~

Uses elliptical constraints for improved reconstruction:

.. code-block:: python

   from AnalysisG.modules.nusol.ellipse import EllipseNuSol
   
   solver = EllipseNuSol()
   # Configure and solve
   solutions = solver.Solve(lepton, met, b_jet)

Conuix Method
~~~~~~~~~~~~~

Conic section intersection approach:

.. code-block:: python

   from AnalysisG.modules.nusol.conuix import ConuixNuSol
   
   solver = ConuixNuSol()
   solutions = solver.Solve(lepton, met, b_jet, top_mass=173.0)

Multiple Solution Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many events have multiple mathematically valid solutions. The module provides:

.. code-block:: python

   from AnalysisG.modules.nusol import MultiSol
   
   solver = MultiSol()
   solutions = solver.SolveAll(event)
   
   # Select best solution
   best_sol = solver.SelectBest(solutions, selection_criteria)

API Reference
-------------

NuSol Class
~~~~~~~~~~~

.. class:: NuSol

   Standard neutrino reconstruction solver.

   **Properties**

   .. attribute:: lepton_pt
      :type: float

      Lepton transverse momentum [GeV].

   .. attribute:: lepton_eta
      :type: float

      Lepton pseudorapidity.

   .. attribute:: lepton_phi
      :type: float

      Lepton azimuthal angle.

   .. attribute:: lepton_E
      :type: float

      Lepton energy [GeV].

   .. attribute:: met
      :type: float

      Missing transverse energy [GeV].

   .. attribute:: met_phi
      :type: float

      MET azimuthal angle.

   .. attribute:: w_mass
      :type: float

      W boson mass constraint [GeV]. Default: 80.4

   .. attribute:: top_mass
      :type: float

      Top quark mass constraint [GeV]. Default: 173.0

   **Methods**

   .. method:: Solve() -> list

      Solve for neutrino solutions.

      :return: List of neutrino solution objects
      :rtype: list

   .. method:: SetMassConstraints(w_mass: float, top_mass: float)

      Set mass constraints.

      :param w_mass: W boson mass [GeV]
      :param top_mass: Top quark mass [GeV]

Solution Object
~~~~~~~~~~~~~~~

.. class:: NuSolSolution

   Represents a neutrino solution.

   **Properties**

   .. attribute:: pt
      :type: float

      Neutrino transverse momentum [GeV].

   .. attribute:: eta
      :type: float

      Neutrino pseudorapidity.

   .. attribute:: phi
      :type: float

      Neutrino azimuthal angle.

   .. attribute:: E
      :type: float

      Neutrino energy [GeV].

   .. attribute:: pz
      :type: float

      Neutrino longitudinal momentum [GeV].

   .. attribute:: chi2
      :type: float

      Chi-squared goodness-of-fit.

   **Methods**

   .. method:: FourVector() -> tuple

      Get 4-vector representation.

      :return: (pt, eta, phi, E) tuple
      :rtype: tuple

Advanced Usage
--------------

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.modules.nusol import NuSol
   
   solver = NuSol()
   
   all_solutions = []
   for event in events:
       solutions = solver.Solve(
           event.lepton, 
           event.met, 
           event.met_phi
       )
       all_solutions.append(solutions)

Solution Selection
~~~~~~~~~~~~~~~~~~

Choose the best solution from multiple candidates:

.. code-block:: python

   def select_best_solution(solutions):
       # Select solution with lowest chi2
       best = min(solutions, key=lambda s: s.chi2)
       return best
   
   # Or use physics constraints
   def select_physical_solution(solutions):
       # Require neutrino |eta| < 5
       physical = [s for s in solutions if abs(s.eta) < 5]
       if not physical:
           return None
       return min(physical, key=lambda s: s.chi2)

C++ Implementation
------------------

The neutrino reconstruction algorithms are implemented in C++ for performance:

- ``nusol/cxx/nusol.cxx``: Core algorithm
- ``ellipse/cxx/ellipse.cxx``: Ellipse method
- ``conuix/cxx/conuix.cxx``: Conic intersection method

The C++ code uses:

- Eigen for linear algebra
- Numerical solvers for nonlinear equations
- Optimized matrix operations

Performance
-----------

The C++ implementation provides:

- ~1-10 μs per event (depending on method)
- Support for batch processing
- CUDA acceleration available (via PyC package)

See Also
--------

* :doc:`../pyc/nusol` - CUDA-accelerated NuSol in PyC
* :doc:`../core/templates` - Template classes
