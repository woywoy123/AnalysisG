enums.h
=======

**File Path**: ``modules/structs/include/structs/enums.h``

**File Type**: H (Header)

**Lines**: 94

Classes
-------

``data_enum``
~~~~~~~~~~~~~

``opt_enum``
~~~~~~~~~~~~

``mlp_init``
~~~~~~~~~~~~

``loss_enum``
~~~~~~~~~~~~~

``scheduler_enum``
~~~~~~~~~~~~~~~~~~

``graph_enum``
~~~~~~~~~~~~~~

``mode_enum``
~~~~~~~~~~~~~

``particle_enum``
~~~~~~~~~~~~~~~~~

Enumerations
------------

``data_enum``

Values: ``// vector<vector<vector<...>>> -> vvv_<X>
// vector<vector<...>> -> vv_<X>
// vector<...> -> v_<X>
// primitives (float``, ``double``, ``long``, ``...)
    d``, ``v_d``, ``vv_d``, ``vvv_d``, ``f``, ``v_f``, ``vv_f``

``opt_enum``

Values: ``adam``, ``adagrad``, ``adamw``, ``lbfgs``, ``rmsprop``, ``sgd``, ``invalid_optimizer``

``mlp_init``

Values: ``uniform``, ``normal``, ``xavier_normal``, ``xavier_uniform``, ``kaiming_uniform``, ``kaiming_normal``

``loss_enum``

Values: ``bce``, ``bce_with_logits``, ``cosine_embedding``, ``cross_entropy``, ``ctc``, ``hinge_embedding``, ``huber``, ``kl_div``, ``l1``, ``margin_ranking``

``scheduler_enum``

Values: ``steplr``, ``reducelronplateauscheduler``, ``lrscheduler``, ``invalid_scheduler``

``graph_enum``

Values: ``data_graph``, ``data_node``, ``data_edge``, ``truth_graph``, ``truth_node``, ``truth_edge``, ``edge_index``, ``weight``, ``batch_index``, ``batch_events``

``mode_enum``

Values: ``training``, ``validation``, ``evaluation``

``particle_enum``

Values: ``index``, ``pdgid``, ``pt``, ``eta``, ``phi``, ``energy``, ``px``, ``pz``, ``py``, ``mass``

