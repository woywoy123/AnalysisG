Inheritance overview
====================

This page summarizes the high-level inheritance structure across the C++ core
and Cython wrappers.

C++ hierarchy (high level)
--------------------------

.. graphviz::

   digraph cpp_hierarchy {
     rankdir=LR;
     node [shape=box];

     tools -> event_template;
     tools -> particle_template;
     tools -> graph_template;
     tools -> selection_template;

     notification -> model_template;
     tools -> model_template;

     notification -> metric_template;
     tools -> metric_template;

     plotting -> roc;

     event_template -> bsm_4tops;
     event_template -> exp_mc20;
     event_template -> gnn_event;
     event_template -> ssml_mc20;

     particle_template -> top;
     particle_template -> child;
     particle_template -> jet;
     particle_template -> electron;
     particle_template -> muon;
     particle_template -> neutrino;

     graph_template -> graph_tops;
     graph_template -> graph_jets;
     graph_template -> graph_detector;

     selection_template -> regions;
     selection_template -> met;
     selection_template -> topmatching;
     selection_template -> multisol;

     model_template -> recyclx;
     model_template -> grift;
     model_template -> recursivegraphneuralnetwork;

     metric_template -> accuracy_metric;
     metric_template -> pagerank_metric;
     metric_template -> topefficiency_metric;
   }

Cython hierarchy (high level)
-----------------------------

.. graphviz::

   digraph cython_hierarchy {
     rankdir=LR;
     node [shape=box];

     EventTemplate -> BSM4Tops;
     EventTemplate -> ExpMC20;
     EventTemplate -> EventGNN;
     EventTemplate -> SSML_MC20;

     ParticleTemplate -> Top;
     ParticleTemplate -> Child;
     ParticleTemplate -> PhysicsDetector;
     ParticleTemplate -> PhysicsTruth;
     ParticleTemplate -> Electron;
     ParticleTemplate -> Muon;
     ParticleTemplate -> Jet;
     ParticleTemplate -> Neutrino;

     GraphTemplate -> GraphTops;
     GraphTemplate -> GraphJets;
     GraphTemplate -> GraphDetector;

     SelectionTemplate -> Regions;
     SelectionTemplate -> MET;
     SelectionTemplate -> TopMatching;
     SelectionTemplate -> Validation;

     ModelTemplate -> RecyclX;
     ModelTemplate -> Grift;
     ModelTemplate -> RecursiveGraphNeuralNetwork;

     MetricTemplate -> AccuracyMetric;
     MetricTemplate -> PageRankMetric;
     MetricTemplate -> TopEfficiencyMetric;

     BasePlotting -> TH1F;
     BasePlotting -> TH2F;
     BasePlotting -> TLine;
     TLine -> ROC;
   }

For full per-module details, see :doc:`cpp/index` and :doc:`cython/index`.
