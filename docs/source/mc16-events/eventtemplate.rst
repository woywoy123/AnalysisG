MC16 BSM-4Tops Event
--------------------
The event implementation for TOPQ1 MC16 n-tuples produced via **AnalysisTop**.

To use this event implementation via Python, use the following import `from AnalysisG.events.event_bsm_4tops import BSM4Tops`.

Event Attributes
^^^^^^^^^^^^^^^^

.. cpp:class:: bsm_4tops: public event_template

   .. cpp:var:: std::vector<particle_template*> Tops

      A vector containing the top-partons casted as its parent class.

   .. cpp:var:: std::vector<particle_template*> Children

      A vector containing the top-parton children casted as the parent class.

   .. cpp:var:: std::vector<particle_template*> TruthJets

      A vector containing the event's truth jets.

   .. cpp:var:: std::vector<particle_template*> Jets

      A vector containing the event's reconstructed jets.

   .. cpp:var:: std::vector<particle_template*> Electrons

      A vector containing the event's electrons.

   .. cpp:var:: std::vector<particle_template*> Muons

      A vector containing the event's muons

   .. cpp:var:: std::vector<particle_template*> DetectorObjects

      A vector containing the detector objects (jets, muon, electrons).

   .. cpp:var:: unsigned long long event_number

      The event number from the n-tuple.

   .. cpp:var:: float mu

      The pile-up of the event.

   .. cpp:var:: float met

      The missing transverse momenta of the event.

   .. cpp:var:: float phi

      The phi of the missing transverse momenta for the given event.
