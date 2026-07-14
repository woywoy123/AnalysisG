Example MET Selection (``selections.example.met``)
===================================================

A minimal worked example showing how to write a selection.  Import with::

    from AnalysisG.selections.example.met import MET

MET
---

``MET`` is a :class:`~AnalysisG.core.selection_template.SelectionTemplate`
subclass that records the missing-ET value for every event that passes
the selection.

**Output attributes** (available after the analysis run):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``missing_et``
     - ``dict[str, float]``
     - Maps event hash string to the MET value [GeV] for each passing
       event.

The underlying C++ class (``met`` in ``met.h``) stores
``map<string, float> missing_et``; the Cython ``transform_dict_keys``
method converts it to a Python ``dict`` after each event via
``as_basic_dict``.
