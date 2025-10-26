Single Neutrino
===============
For this study the Monte Carlo truth is varied, the first set of plots illustrate the possible resolution achievable using truth children, in particular truth b-quark and lepton, which are shown below as under Figure.1 collections.
As for the Figure.2 collections, the truth b-quark is replaced with a truth-jet matched to the b-quark parton.
In cases where the b-quark has no associated truth-jet, the event is skipped. 
Figure.3 collections use the detector reconstructed jets but still use the truth children lepton.
Again for instances where no jet is matched to the associated b-quark, the event is skipped.
Finally, Figure.4 collections illustrate the worst possible resolution, since the truth children lepton is replaced with detector based matched lepton.
Similar to the (truth)-jet cases, if neither a b-quark has been matched to a jet or the truth children lepton has no associated matching, the event is skipped.

The latter part of the study focuses on bruteforcing the S-matrix values, which minimize the :math:`\chi` value of the difference between the truth and reconstructed neutrino kinematic values.

Selection Criteria
------------------
Events are required to have exactly one top-quark decaying leptonically.
The :math:`E_T` and :math:`\phi` of the detector is used as input for the algorithm, with the top and W masses being derived from the truth children or matched (truth)-jets and detector leptons.
The neutrino algorithm **zero** value is being set to 1e-10, since anything below this value creates large differences between C++ and Python floating point definitions.
Furthermore, the momentum imbalance uncertainty matrix **S**, has been assigned the values: :math:`S_xx = 1000, S_xy = 100, S_yx = 100, S_yy = 1000`. 
These values were chosen arbitrarily, and not optimized for the bulk of the study.
For the S-Matrix optimization, only detector based jets and leptons are used.


Particle Definitions
--------------------
Leptons and neutrinos are defined as:

- electrons
- muons 
- taus

**Truth Children**
------------------

Figure 1.a (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.a.png
   :align: center
   :name: Figure.1.nu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 1.b (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.b.png
   :align: center
   :name: Figure.1.nu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 1.c (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.c.png
   :align: center
   :name: Figure.1.nu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 1.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.d.png
   :align: center
   :name: Figure.1.nu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 1.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.e.png
   :align: center
   :name: Figure.1.nu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 1.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.f.png
   :align: center
   :name: Figure.1.nu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 1.g
^^^^^^^^^^
.. figure:: ./figures/Figure.1.g.png
   :align: center
   :name: Figure.1.nu.g

   A projection plot in the :math:`P_x` direction illustrating differences between the reference and pyc implementions.

Figure 1.h
^^^^^^^^^^
.. figure:: ./figures/Figure.1.h.png
   :align: center
   :name: Figure.1.nu.h

   A projection plot in the :math:`P_y` direction illustrating differences between the reference and pyc implementions.

Figure 1.i
^^^^^^^^^^
.. figure:: ./figures/Figure.1.i.png
   :align: center
   :name: Figure.1.nu.i

   A projection plot in the :math:`P_z` direction illustrating differences between the reference and pyc implementions.

Figure 1.j
^^^^^^^^^^
.. figure:: ./figures/Figure.1.j.png
   :align: center
   :name: Figure.1.nu.j

   A plot illustrating the energy difference between the truth and reconstructed neutrino for the reference and pyc implementation.

Figure 1.k
^^^^^^^^^^
.. figure:: ./figures/Figure.1.k.png
   :align: center
   :name: Figure.1.nu.k

   Reconstructed invariant top-mass using the reference and pyc implementations, compared to the true top-mass parton mass.



**Truth Jets**
--------------

Figure 2.a (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.a.png
   :align: center
   :name: Figure.2.nu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 2.b (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.b.png
   :align: center
   :name: Figure.2.nu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 2.c (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.c.png
   :align: center
   :name: Figure.2.nu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 2.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.d.png
   :align: center
   :name: Figure.2.nu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 2.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.e.png
   :align: center
   :name: Figure.2.nu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 2.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.f.png
   :align: center
   :name: Figure.2.nu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 2.g
^^^^^^^^^^
.. figure:: ./figures/Figure.2.g.png
   :align: center
   :name: Figure.2.nu.g

   A projection plot in the :math:`P_x` direction illustrating differences between the reference and pyc implementions.

Figure 2.h
^^^^^^^^^^
.. figure:: ./figures/Figure.2.h.png
   :align: center
   :name: Figure.2.nu.h

   A projection plot in the :math:`P_y` direction illustrating differences between the reference and pyc implementions.

Figure 2.i
^^^^^^^^^^
.. figure:: ./figures/Figure.2.i.png
   :align: center
   :name: Figure.2.nu.i

   A projection plot in the :math:`P_z` direction illustrating differences between the reference and pyc implementions.

Figure 2.j
^^^^^^^^^^
.. figure:: ./figures/Figure.2.j.png
   :align: center
   :name: Figure.2.nu.j

   A plot illustrating the energy difference between the truth and reconstructed neutrino for the reference and pyc implementation.

Figure 2.k
^^^^^^^^^^
.. figure:: ./figures/Figure.2.k.png
   :align: center
   :name: Figure.2.nu.k

   Reconstructed invariant top-mass using the reference and pyc implementations, compared to the true top-mass parton mass.


**Jets**
--------

Figure 3.a (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.a.png
   :align: center
   :name: Figure.3.nu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 3.b (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.b.png
   :align: center
   :name: Figure.3.nu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 3.c (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.c.png
   :align: center
   :name: Figure.3.nu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 3.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.d.png
   :align: center
   :name: Figure.3.nu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 3.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.e.png
   :align: center
   :name: Figure.3.nu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 3.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.f.png
   :align: center
   :name: Figure.3.nu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 3.g
^^^^^^^^^^
.. figure:: ./figures/Figure.3.g.png
   :align: center
   :name: Figure.3.nu.g

   A projection plot in the :math:`P_x` direction illustrating differences between the reference and pyc implementions.

Figure 3.h
^^^^^^^^^^
.. figure:: ./figures/Figure.3.h.png
   :align: center
   :name: Figure.3.nu.h

   A projection plot in the :math:`P_y` direction illustrating differences between the reference and pyc implementions.

Figure 3.i
^^^^^^^^^^
.. figure:: ./figures/Figure.3.i.png
   :align: center
   :name: Figure.3.nu.i

   A projection plot in the :math:`P_z` direction illustrating differences between the reference and pyc implementions.

Figure 3.j
^^^^^^^^^^
.. figure:: ./figures/Figure.3.j.png
   :align: center
   :name: Figure.3.nu.j

   A plot illustrating the energy difference between the truth and reconstructed neutrino for the reference and pyc implementation.

Figure 3.k
^^^^^^^^^^
.. figure:: ./figures/Figure.3.k.png
   :align: center
   :name: Figure.3.nu.k

   Reconstructed invariant top-mass using the reference and pyc implementations, compared to the true top-mass parton mass.


**Jets with Detector Leptons**
------------------------------

Figure 4.a (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.a.png
   :align: center
   :name: Figure.4.nu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 4.b (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.b.png
   :align: center
   :name: Figure.4.nu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 4.c (PYC)
^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.c.png
   :align: center
   :name: Figure.4.nu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino (using **pyc**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 4.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.d.png
   :align: center
   :name: Figure.4.nu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 4.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.e.png
   :align: center
   :name: Figure.4.nu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 4.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.f.png
   :align: center
   :name: Figure.4.nu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino (using **reference**).
   This is to validate whether the difference between the truth and the reconstructed neutrino is consistent. 
   Ideally the heat-map should have a single bin at (0,0), indicating perfect reconstruction.

Figure 4.g
^^^^^^^^^^
.. figure:: ./figures/Figure.4.g.png
   :align: center
   :name: Figure.4.nu.g

   A projection plot in the :math:`P_x` direction illustrating differences between the reference and pyc implementions.

Figure 4.h
^^^^^^^^^^
.. figure:: ./figures/Figure.4.h.png
   :align: center
   :name: Figure.4.nu.h

   A projection plot in the :math:`P_y` direction illustrating differences between the reference and pyc implementions.

Figure 4.i
^^^^^^^^^^
.. figure:: ./figures/Figure.4.i.png
   :align: center
   :name: Figure.4.nu.i

   A projection plot in the :math:`P_z` direction illustrating differences between the reference and pyc implementions.

Figure 4.j
^^^^^^^^^^
.. figure:: ./figures/Figure.4.j.png
   :align: center
   :name: Figure.4.nu.j

   A plot illustrating the energy difference between the truth and reconstructed neutrino for the reference and pyc implementation.

Figure 4.k
^^^^^^^^^^
.. figure:: ./figures/Figure.4.k.png
   :align: center
   :name: Figure.4.nu.k

   Reconstructed invariant top-mass using the reference and pyc implementations, compared to the true top-mass parton mass.


S-Matrix (Momentum Imbalance Uncertainty Matrix)
------------------------------------------------

.. figure:: ./figures/Figure.5.a.png
   :align: center
   :name: Figure.5.nu.a

   A heat-map of the S-matrix values for the diagonal and non-diagonal elements.
   Each bin counts the frequency of finding the lowest :math:`\chi` at the given matrix values.
   Large clustering around particular :math:`(S_{xx}, S_{yy}), (S_{xy}, S_{yx})` pairs indicates a potential optimization point.
   However, if no clustering is observed, the S-matrix values can be chosen arbitrarily, with no preferential values.
   Note: This is for the **pyc** implementation.

.. figure:: ./figures/Figure.5.b.png
   :align: center
   :name: Figure.5.nu.b

   A heat-map of the S-matrix values for the diagonal and non-diagonal elements.
   Each bin counts the frequency of finding the lowest :math:`\chi` at the given matrix values.
   Large clustering around particular :math:`(S_{xx}, S_{yy}), (S_{xy}, S_{yx})` pairs indicates a potential optimization point.
   However, if no clustering is observed, the S-matrix values can be chosen arbitrarily, with no preferential values.
   Note: This is for the **reference** implementation.

.. figure:: ./figures/Figure.5.c.png
   :align: center
   :name: Figure.5.nu.c

   A histogram plot depicting the :math:`\Delta Px` of the truth neutrino (truth children) and the reconstructed neutrino.
   In order to compare the performance of the two implementations, clustering around 0 indicates perfect reconstruction.

.. figure:: ./figures/Figure.5.d.png
   :align: center
   :name: Figure.5.nu.d

   A histogram plot depicting the :math:`\Delta Py` of the truth neutrino (truth children) and the reconstructed neutrino.
   In order to compare the performance of the two implementations, clustering around 0 indicates perfect reconstruction.

.. figure:: ./figures/Figure.5.e.png
   :align: center
   :name: Figure.5.nu.e

   A histogram plot depicting the :math:`\Delta Pz` of the truth neutrino (truth children) and the reconstructed neutrino.
   In order to compare the performance of the two implementations, clustering around 0 indicates perfect reconstruction.

.. figure:: ./figures/Figure.5.f.png
   :align: center
   :name: Figure.5.nu.f

   A histogram plot depicting the :math:`\chi` of summing the difference between the individual momentum components of the truth and reconstructed neutrino.
   Values close to 0, indicate better reconstruction performance.


