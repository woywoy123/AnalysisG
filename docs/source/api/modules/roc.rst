ROC Curve Module
================

The ``roc`` class derives from ``plotting`` and provides ROC (Receiver Operating
Characteristic) curve computation and storage for multi-class classifiers.
It is used by the built-in ``AccuracyMetric`` to generate AUC tables after training.

Struct: ``roc_t``
-----------------

**Header:** ``<plotting/roc.h>``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Field
     - Type
     - Description
   * - ``cls``
     - ``int``
     - Class index (0-based) that this ROC curve belongs to.
   * - ``kfold``
     - ``int``
     - k-fold index for which the ROC was computed.
   * - ``model``
     - ``std::string``
     - Name of the model that produced these scores.
   * - ``_auc``
     - ``std::vector<double>``
     - Area under the ROC curve for each threshold (one per class).
   * - ``tpr_``
     - ``std::vector<std::vector<double>>``
     - True-positive rates across thresholds (outer: threshold, inner: class).
   * - ``fpr_``
     - ``std::vector<std::vector<double>>``
     - False-positive rates across thresholds.
   * - ``truth``
     - ``std::vector<std::vector<int>>*``
     - Pointer to truth labels (shape ``[N_events, N_classes]``).
   * - ``scores``
     - ``std::vector<std::vector<double>>*``
     - Pointer to model score tensors (shape ``[N_events, N_classes]``).

Class: ``roc``
--------------

**Header:** ``<plotting/roc.h>``

**Inheritance:** ``plotting``

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``void build_ROC(std::string name, int kfold, std::vector<int>* label, std::vector<std::vector<double>>* scores)``
     - Computes the ROC curves for classifier *name*, fold *kfold*, given
       ground-truth *label* (flat integer class vector) and *scores* (one
       row per event, one column per class).  Results are stored in the
       internal ``roc_data`` and ``labels`` maps.
   * - ``std::vector<roc_t*> get_ROC()``
     - Returns pointers to all computed ``roc_t`` objects.  The caller does
       **not** own the returned pointers (owned by ``roc``).

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Field / Type
     - Description
   * - ``std::map<std::string, std::map<int, std::vector<std::vector<double>>*>> roc_data``
     - Maps model-name → {kfold → score vectors}.
   * - ``std::map<std::string, std::map<int, std::vector<std::vector<int>>*>> labels``
     - Maps model-name → {kfold → truth label vectors}.

Example::

    roc roc_obj;
    roc_obj.build_ROC("MyModel", 0, &truth_labels, &model_scores);
    for (roc_t* r : roc_obj.get_ROC()) {
        std::cout << "AUC (class 0): " << r->_auc[0] << std::endl;
    }
