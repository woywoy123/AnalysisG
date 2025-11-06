.. _beyond-standard-model: https://github.com/woywoy123/BSM4tops-GNN-ntuples

Same Sign Multi-Lepton Event (MC-20)
------------------------------------

The event implementation is based on `beyond-standard-model`_, which interprets the additional truth matching branches in the output n-tuples.

Event Attribute Descriptions
____________________________


.. py:class:: AnalysisG.Events import SSML_MC20

    :ivar is_mc: "mcChannelNumber"
    :ivar met_sum: "met_sumet"
    :ivar met: "met_met"
    :ivar phi: "met_phi"
    :ivar weight: "weight_mc"
    :ivar mu: "mu"
    :ivar Tops: A list of final state radiation top quarks in the event.
    :ivar TruthChildren: A list of all truth children originating from the top partons.
    :ivar PhysicsTruth: A list of all truth-jets and associated electron and muons. 
    :ivar Jets: A list of event Jets.
    :ivar Leptons: A cummulated list of muons and electrons in the event
    :ivar Detector: A list of all detector objects (jets/electrons/muons/photons).

Particle Definitions
____________________

.. py:class:: AnalysisG.Events.Events.mc20_ssml.Particles.Top

    :ivar Type: "top"
    :ivar pt: "top_pt"
    :ivar eta: "top_eta"
    :ivar phi: "top_phi"
    :ivar e: "top_e"
    :ivar index: "top_top_index" - A unique top index for the set of tops in the event.
    :ivar charge: "top_charge" - Assigned charge of the top parton
    :ivar barcode: "top_barcode" - The barcode assigned to a given top by AnalysisTop
    :ivar pdgid: "top_pdgid" - The pdgid of the given top.
    :ivar status: "top_status" - The decay status of the final top quark.
    :ivar Children: A container used to link the truth children to the top quark.

.. py:class:: AnalysisG.Events.Events.mc20_ssml.Particles.Children

    :ivar Type: "child"
    :ivar pt: "child_pt"
    :ivar eta: "child_eta"
    :ivar phi: "child_phi"
    :ivar e: "child_e"
    :ivar index: "child_top_index" - The assigned top index, that the child belongs to.
    :ivar barcode: "child_barcode" - The assigned barcode of the child object (from AnalysisTop).
    :ivar charge: "child_charge" - Assigned charge of the child object.
    :ivar pdgid: "child_pdgid" - The pdgid of the given object.
    :ivar status: "child_status" - The Pythia status code of the particle.
    :ivar Parent: A list containing the originating top quark.

.. py:class:: AnalysisG.Events.Events.mc20_ssml.Particles.PhysicsDetector

    :ivar Type: "physdet"
    :ivar pt: "physdet_pt"
    :ivar eta: "physdet_eta"
    :ivar phi: "physdet_phi"
    :ivar e: "physdet_e"
    :ivar index: "physdet_index"
    :ivar top_index: "physdet_top_index"
    :ivar truth_parton: "physdet_partontruthlabel"
    :ivar particle_type: "physdet_type" ([jet, lep, photon])
    :ivar truth_cone: "physdet_contruthlabel"
    
.. py:class:: AnalysisG.Events.Events.mc20_ssml.Particles.PhysicsTruth

    :ivar Type: "phystru"
    :ivar pt: "phystru_pt"
    :ivar eta: "phystru_eta"
    :ivar phi: "phystru_phi"
    :ivar e: "phystru_e"
    :ivar index: "phystru_index"
    :ivar top_index: "phystru_top_index"
    :ivar truth_parton: "phystru_partontruthlabel"
    :ivar particle_type: "phystru_type" ([jet, lep, photon])
    :ivar truth_cone: "phystru_contruthlabel"

.. py:class:: AnalysisG.Events.Events.mc20_ssml.Particles.Electron

    :ivar Type: "el"
    :ivar pt: "el_pt"
    :ivar eta: "el_eta"
    :ivar phi: "el_phi"
    :ivar e: "el_e"
    :ivar charge: "el_charge"
    :ivar tight: "el_isTight"
    :ivar d0sig: "el_d0sig"
    :ivar delta_z0: "el_delta_z0_sintheta"
    :ivar true_type: "el_true_type"
    :ivar true_origin: "el_true_origin"

.. py:class:: AnalysisG.Events.Events.mc20_ssml.Particles.Muon

    :ivar Type: "mu"
    :ivar pt: "mu_pt"
    :ivar eta: "mu_eta"
    :ivar phi: "mu_phi"
    :ivar e: "mu_e"
    :ivar charge: "mu_charge"
    :ivar tight: "mu_isTight"
    :ivar d0sig: "mu_d0sig"
    :ivar delta_z0: "mu_delta_z0_sintheta"
    :ivar true_type: "mu_true_type"
    :ivar true_origin: "mu_true_origin"

.. py:class:: AnalysisG.Events.Events.mc20_ssml.Particles.Jet

    :ivar Type: "jet"
    :ivar pt: "jet_pt"
    :ivar eta: "jet_eta"
    :ivar phi: "jet_phi"
    :ivar e: "jet_e"
    :ivar jvt: "jet_jvt"
    :ivar truth_flavor: "jet_truthflav"
    :ivar truth_parton: "jet_truthPartonLabel"
    :ivar btag60: "jet_isbtagged_GN2v00NewAliasWP_60"
    :ivar btag70: "jet_isbtagged_GN2v00NewAliasWP_70"
    :ivar btag77: "jet_isbtagged_GN2v00NewAliasWP_77"
    :ivar btag85: "jet_isbtagged_GN2v00NewAliasWP_85"





