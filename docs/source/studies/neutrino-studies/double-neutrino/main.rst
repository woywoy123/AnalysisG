Double Neutrino
===============
For this study, the double neutrino algorithm is being validated against the reference implementation using varying levels of Monte Carlo truth.
Similar to the single neutrino reconstruction, the aim is to assert whether the custom implementation is consistent and how well it reconstructs the neutrino pairs.
Starting with the simplest case, the truth children b-parton and lepton pairs are given to the algorithm to compare the resultant kinematics to the true neutrinos.
This is followed by the replacement of the b-parton with matched truth jets and detector jets.
Finally, only detector objects will be used to perform the reconstruction, this includes matched detector jets and leptons.

Selection Criteria
------------------
Events are required to have exactly two leptonically decaying top-quarks on truth children level.
For subsequent studies involving truth jets and detector based objects, both b-quarks and leptons on truth children level need to be matched accordingly.
If any of these conditions are not satisfied, the respective part of the study is skipped for the event.
Furthermore, truth jets and jets are required to have only single top contributions, otherwise the event is vetoed.

Particle Definitions
--------------------
Leptons and neutrinos are defined as:

- electrons
- muons
- taus

**Truth Children**
------------------

Figure 1.1.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.1.a.png
   :align: center
   :name: Figure.1.1.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.2.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.2.a.png
   :align: center
   :name: Figure.1.2.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.3.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.3.a.png
   :align: center
   :name: Figure.1.3.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 1.1.a and 1.2.a. 


Figure 1.1.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.1.b.png
   :align: center
   :name: Figure.1.1.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.2.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.2.b.png
   :align: center
   :name: Figure.1.2.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.3.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.3.b.png
   :align: center
   :name: Figure.1.3.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 1.1.b and 1.2.b. 

Figure 1.1.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.1.c.png
   :align: center
   :name: Figure.1.1.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.2.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.2.c.png
   :align: center
   :name: Figure.1.2.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.3.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.3.c.png
   :align: center
   :name: Figure.1.3.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 1.1.c and 1.2.c. 

Figure 1.1.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.1.d.png
   :align: center
   :name: Figure.1.1.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.2.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.2.d.png
   :align: center
   :name: Figure.1.2.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.3.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.3.d.png
   :align: center
   :name: Figure.1.3.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 1.1.d and 1.2.d. 


Figure 1.1.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.1.e.png
   :align: center
   :name: Figure.1.1.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.2.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.2.e.png
   :align: center
   :name: Figure.1.2.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.3.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.3.e.png
   :align: center
   :name: Figure.1.3.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 1.1.e and 1.2.e. 

Figure 1.1.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.1.f.png
   :align: center
   :name: Figure.1.1.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.2.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.2.f.png
   :align: center
   :name: Figure.1.2.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 1.3.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.1.3.f.png
   :align: center
   :name: Figure.1.3.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 1.1.f and 1.2.f. 

Figure 1.g
^^^^^^^^^^
.. figure:: ./figures/Figure.1.g.png
   :align: center
   :name: Figure.1.nunu.g

   A projection plot in the :math:`P_x` direction illustrating differences between the reference and pyc implementions.

Figure 1.h
^^^^^^^^^^
.. figure:: ./figures/Figure.1.h.png
   :align: center
   :name: Figure.1.nunu.h

   A projection plot in the :math:`P_y` direction illustrating differences between the reference and pyc implementions.

Figure 1.i
^^^^^^^^^^
.. figure:: ./figures/Figure.1.i.png
   :align: center
   :name: Figure.1.nunu.i

   A projection plot in the :math:`P_z` direction illustrating differences between the reference and pyc implementions.

Figure 1.j
^^^^^^^^^^
.. figure:: ./figures/Figure.1.j.png
   :align: center
   :name: Figure.1.nunu.j

   A plot illustrating the energy difference between the truth and reconstructed neutrino for the reference and pyc implementation.

Figure 1.k
^^^^^^^^^^
.. figure:: ./figures/Figure.1.k.png
   :align: center
   :name: Figure.1.nunu.k

   Reconstructed invariant top-mass using the reference and pyc implementations, compared to the true top-mass parton mass.


**Truth Jets**
--------------

Figure 2.1.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.1.a.png
   :align: center
   :name: Figure.2.1.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.2.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.2.a.png
   :align: center
   :name: Figure.2.2.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.3.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.3.a.png
   :align: center
   :name: Figure.2.3.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 2.1.a and 2.2.a. 


Figure 2.1.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.1.b.png
   :align: center
   :name: Figure.2.1.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.2.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.2.b.png
   :align: center
   :name: Figure.2.2.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.3.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.3.b.png
   :align: center
   :name: Figure.2.3.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 2.1.b and 2.2.b. 

Figure 2.1.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.1.c.png
   :align: center
   :name: Figure.2.1.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.2.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.2.c.png
   :align: center
   :name: Figure.2.2.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.3.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.3.c.png
   :align: center
   :name: Figure.2.3.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 2.1.c and 2.2.c. 

Figure 2.1.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.1.d.png
   :align: center
   :name: Figure.2.1.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.2.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.2.d.png
   :align: center
   :name: Figure.2.2.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.3.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.3.d.png
   :align: center
   :name: Figure.2.3.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 2.1.d and 2.2.d. 


Figure 2.1.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.1.e.png
   :align: center
   :name: Figure.2.1.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.2.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.2.e.png
   :align: center
   :name: Figure.2.2.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.3.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.3.e.png
   :align: center
   :name: Figure.2.3.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 2.1.e and 2.2.e. 

Figure 2.1.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.1.f.png
   :align: center
   :name: Figure.2.1.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.2.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.2.f.png
   :align: center
   :name: Figure.2.2.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 2.3.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.2.3.f.png
   :align: center
   :name: Figure.2.3.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 2.1.f and 2.2.f. 

Figure 2.g
^^^^^^^^^^
.. figure:: ./figures/Figure.2.g.png
   :align: center
   :name: Figure.2.nunu.g

   A projection plot in the :math:`P_x` direction illustrating differences between the reference and pyc implementions.

Figure 2.h
^^^^^^^^^^
.. figure:: ./figures/Figure.2.h.png
   :align: center
   :name: Figure.2.nunu.h

   A projection plot in the :math:`P_y` direction illustrating differences between the reference and pyc implementions.

Figure 2.i
^^^^^^^^^^
.. figure:: ./figures/Figure.2.i.png
   :align: center
   :name: Figure.2.nunu.i

   A projection plot in the :math:`P_z` direction illustrating differences between the reference and pyc implementions.

Figure 2.j
^^^^^^^^^^
.. figure:: ./figures/Figure.2.j.png
   :align: center
   :name: Figure.2.nunu.j

   A plot illustrating the energy difference between the truth and reconstructed neutrino for the reference and pyc implementation.

Figure 2.k
^^^^^^^^^^
.. figure:: ./figures/Figure.2.k.png
   :align: center
   :name: Figure.2.nunu.k

   Reconstructed invariant top-mass using the reference and pyc implementations, compared to the true top-mass parton mass.


**Jets**
--------

Figure 3.1.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.1.a.png
   :align: center
   :name: Figure.3.1.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.2.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.2.a.png
   :align: center
   :name: Figure.3.2.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.3.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.3.a.png
   :align: center
   :name: Figure.3.3.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 3.1.a and 3.2.a. 


Figure 3.1.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.1.b.png
   :align: center
   :name: Figure.3.1.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.2.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.2.b.png
   :align: center
   :name: Figure.3.2.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.3.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.3.b.png
   :align: center
   :name: Figure.3.3.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 3.1.b and 3.2.b. 

Figure 3.1.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.1.c.png
   :align: center
   :name: Figure.3.1.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.2.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.2.c.png
   :align: center
   :name: Figure.3.2.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.3.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.3.c.png
   :align: center
   :name: Figure.3.3.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 3.1.c and 3.2.c. 

Figure 3.1.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.1.d.png
   :align: center
   :name: Figure.3.1.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.2.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.2.d.png
   :align: center
   :name: Figure.3.2.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.3.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.3.d.png
   :align: center
   :name: Figure.3.3.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 3.1.d and 3.2.d. 


Figure 3.1.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.1.e.png
   :align: center
   :name: Figure.3.1.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.2.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.2.e.png
   :align: center
   :name: Figure.3.2.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.3.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.3.e.png
   :align: center
   :name: Figure.3.3.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 3.1.e and 3.2.e. 

Figure 3.1.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.1.f.png
   :align: center
   :name: Figure.3.1.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.2.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.2.f.png
   :align: center
   :name: Figure.3.2.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure 3.3.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.3.3.f.png
   :align: center
   :name: Figure.3.3.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 3.1.f and 3.2.f. 

Figure 3.g
^^^^^^^^^^
.. figure:: ./figures/Figure.3.g.png
   :align: center
   :name: Figure.3.nunu.g

   A projection plot in the :math:`P_x` direction illustrating differences between the reference and pyc implementions.

Figure 3.h
^^^^^^^^^^
.. figure:: ./figures/Figure.3.h.png
   :align: center
   :name: Figure.3.nunu.h

   A projection plot in the :math:`P_y` direction illustrating differences between the reference and pyc implementions.

Figure 3.i
^^^^^^^^^^
.. figure:: ./figures/Figure.3.i.png
   :align: center
   :name: Figure.3.nunu.i

   A projection plot in the :math:`P_z` direction illustrating differences between the reference and pyc implementions.

Figure 3.j
^^^^^^^^^^
.. figure:: ./figures/Figure.3.j.png
   :align: center
   :name: Figure.3.nunu.j

   A plot illustrating the energy difference between the truth and reconstructed neutrino for the reference and pyc implementation.

Figure 3.k
^^^^^^^^^^
.. figure:: ./figures/Figure.3.k.png
   :align: center
   :name: Figure.3.nunu.k

   Reconstructed invariant top-mass using the reference and pyc implementations, compared to the true top-mass parton mass.



**Jets with Detector Leptons**
------------------------------

Figure.4.1.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.1.a.png
   :align: center
   :name: Figure.4.1.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.2.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.2.a.png
   :align: center
   :name: Figure.4.2.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.3.a (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.3.a.png
   :align: center
   :name: Figure.4.3.nunu.a

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 4.1.a and 4.2.a. 


Figure.4.1.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.1.b.png
   :align: center
   :name: Figure.4.1.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.2.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.2.b.png
   :align: center
   :name: Figure.4.2.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.3.b (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.3.b.png
   :align: center
   :name: Figure.4.3.nunu.b

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 4.1.b and 4.2.b. 

Figure.4.1.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.1.c.png
   :align: center
   :name: Figure.4.1.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.2.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.2.c.png
   :align: center
   :name: Figure.4.2.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **pyc**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.3.c (PYC)
^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.3.c.png
   :align: center
   :name: Figure.4.3.nunu.c

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 4.1.c and 4.2.c. 

Figure.4.1.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.1.d.png
   :align: center
   :name: Figure.4.1.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.2.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.2.d.png
   :align: center
   :name: Figure.4.2.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.3.d (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.3.d.png
   :align: center
   :name: Figure.4.3.nunu.d

   A heat-map of the momenta differential in the x and y direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 4.1.d and 4.2.d. 


Figure.4.1.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.1.e.png
   :align: center
   :name: Figure.4.1.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.2.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.2.e.png
   :align: center
   :name: Figure.4.2.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.3.e (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.3.e.png
   :align: center
   :name: Figure.4.3.nunu.e

   A heat-map of the momenta differential in the x and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 4.1.e and 4.2.e. 

Figure.4.1.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.1.f.png
   :align: center
   :name: Figure.4.1.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.2.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.2.f.png
   :align: center
   :name: Figure.4.2.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino.
   In this plot, one of the neutrinos in the reconstructed pair is being compared to its associated truth neutrino.
   The algorithm used to generate these solutions is based on **reference**.
   The purpose of looking at only one of the neutrinos is to determine whether the neutrino solutions are consistent, and if there is an error asymmetry.
   Ideally, the neutrino pairs cluster around the (0, 0) coordinate, indicating a consistent implementation.

Figure.4.3.f (REFERENCE)
^^^^^^^^^^^^^^^^^^^^^^^^
.. figure:: ./figures/Figure.4.3.f.png
   :align: center
   :name: Figure.4.3.nunu.f

   A heat-map of the momenta differential in the y and z direction between the truth and reconstructed neutrino solution pairs.
   This figure is the cummulation of Figure 4 1.f and 4.2.f. 

Figure.4.g
^^^^^^^^^^
.. figure:: ./figures/Figure.4.g.png
   :align: center
   :name: Figure.4.nunu.g

   A projection plot in the :math:`P_x` direction illustrating differences between the reference and pyc implementions.

Figure.4.h
^^^^^^^^^^
.. figure:: ./figures/Figure.4.h.png
   :align: center
   :name: Figure.4.nunu.h

   A projection plot in the :math:`P_y` direction illustrating differences between the reference and pyc implementions.

Figure.4.i
^^^^^^^^^^
.. figure:: ./figures/Figure.4.i.png
   :align: center
   :name: Figure.4.nunu.i

   A projection plot in the :math:`P_z` direction illustrating differences between the reference and pyc implementions.

Figure.4.j
^^^^^^^^^^
.. figure:: ./figures/Figure.4.j.png
   :align: center
   :name: Figure.4.nunu.j

   A plot illustrating the energy difference between the truth and reconstructed neutrino for the reference and pyc implementation.

Figure.4.k
^^^^^^^^^^
.. figure:: ./figures/Figure.4.k.png
   :align: center
   :name: Figure.4.nunu.k

   Reconstructed invariant top-mass using the reference and pyc implementations, compared to the true top-mass parton mass.

