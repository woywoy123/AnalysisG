MC16 BSM-4Tops Particle Definitions
-----------------------------------

Documentation used for the particles defined in the MC16-BSM-4Tops event implementation.


Generator Top Parton
^^^^^^^^^^^^^^^^^^^^

.. cpp:class:: top: public particle_template

   .. cpp:var:: bool from_res 
   
      Specifies whether the generator top parton originates from a heavy scalar boson.

   .. cpp:var:: int status
   
      The status of the top-parton, indicates whether the particle is in its final form (post gluon radiation)

   .. cpp:var:: std::vector<truthjet> TruthJets

      A vector holding matched truth jets.

   .. cpp:var:: std::vector<jet> Jets

      A vector holding matched detector reconstructed jets.


Generator Top Parton Decay Children
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:class:: top_children: public particle_template

   .. cpp:var:: int top_index

      The index of the top parton the child originates from.
      A top child is any particle (except an intermediate boson - e.g. W) that originates from the decay of a top-parton

   .. cpp:var:: cproperty<bool, top_children> from_res
   
      Indicates whether the parent top-parton originates from a heavy scalar boson.


Truth Jet
^^^^^^^^^

.. cpp:class:: truthjet: public particle_template

   .. cpp:var:: int top_quark_count

      Indicates the number of top-partons contributing to the truth jet.
      This value is derived from the n-tuples.

   .. cpp:var:: int w_boson_count

      Indicates the number of w-bosons contributing to the truth jet.
      This value is derived from the n-tuples.

   .. cpp:var:: std::vector<int> top_index

      A vector indicating the top parton index matching to this truth jet.

   .. cpp:var:: std::vector<top> Tops

      A vector holding the matched top quark parton objects.

   .. cpp:var:: std::vector<truthjetparton> Parton

      A vector containing the partons matched to the truth jet.

   .. cpp:var: cproperty<bool, truthjet> from_res

      A variable indicating whether the matched top-partons originate from a heavy boson.


Truth Jet Partons
^^^^^^^^^^^^^^^^^

.. cpp:class:: truthjetparton: public particle_template

   .. cpp:var:: int truthjet_index

      A variable used to match the parton to the contributing truthjet.

   .. cpp:var:: std::vector<int> topchild_index

      A variable containing the index of the top_children that produced the parton.


Detector Reconstructed Jet
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:class:: jet: public particle_template

   .. cpp:var:: std::vector<top> Tops

      A vector containing the top-partons matched to the particular jet.

   .. cpp:var:: std::vector<jetparton> Parton

      A vector containing the partons matched to the jet.

   .. cpp:var:: std::vector<int> top_index

      A vector indictating the top parton index matching to this jet.
 
   .. cpp:var:: float btag_DL1r_60

   .. cpp:var:: float btag_DL1_60

   .. cpp:var:: float btag_DL1r_70

   .. cpp:var:: float btag_DL1_70

   .. cpp:var:: float btag_DL1r_77

   .. cpp:var:: float btag_DL1_77

   .. cpp:var:: float btag_DL1r_85

   .. cpp:var:: float btag_DL1_85

   .. cpp:var:: float DL1_b

   .. cpp:var:: float DL1_c

   .. cpp:var:: float DL1_u

   .. cpp:var:: float DL1r_b

   .. cpp:var:: float DL1r_c

   .. cpp:var:: float DL1r_u

