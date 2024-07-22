.. _particle-struct:

particle_t Struct Attributes
----------------------------

.. cpp:struct:: particle_meta_t

   .. cpp:var:: string children

   Child hash value.

   .. cpp:var:: string parents

   Parent hash value.

   .. cpp:var:: string lepdef

   .. cpp:var:: string nudef


.. cpp:struct:: particle_t 

   .. cpp:var:: double e

   .. cpp:var:: double mass

   .. cpp:var:: double px

   .. cpp:var:: double py

   .. cpp:var:: double pz

   .. cpp:var:: double pt

   .. cpp:var:: double eta

   .. cpp:var:: double phi

   .. cpp:var:: bool cartesian

   .. cpp:var:: bool polar

   .. cpp:var:: double charge

   .. cpp:var:: int pdgid

   .. cpp:var:: int index 

   .. cpp:var:: std::string type

   .. cpp:var:: std::string hash

   .. cpp:var:: std::string symbol

   .. cpp:var:: std::vector<int> lepdef

   .. cpp:var:: std::vector<int> nudef

   .. cpp:var:: std::map<std::string, bool> children

   .. cpp:var:: std::map<std::string, bool> parents

