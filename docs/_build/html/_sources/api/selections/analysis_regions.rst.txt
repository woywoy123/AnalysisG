Analysis Regions Selection (``selections.analysis.regions``)
=============================================================

Import with::

    from AnalysisG.selections.analysis.regions import Regions

Regions
-------

``Regions`` is a :class:`~AnalysisG.core.selection_template.SelectionTemplate`
subclass that evaluates ATLAS analysis regions (Control/Validation/Signal
Regions) for each event.

Each region is represented by a ``regions_t`` struct with fields:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``variable1``
     - ``float``
     - Primary discriminating variable.
   * - ``variable2``
     - ``float``
     - Secondary discriminating variable.
   * - ``weight``
     - ``float``
     - Event weight.
   * - ``passed``
     - ``bool``
     - Whether the event passed the region selection.

**Output attribute**:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``output``
     - ``list[dict]``
     - One ``package_t`` dictionary per event.  Each dictionary maps
       region name to its ``regions_t`` fields.

**Regions defined** in ``package_t``:
``CRttbarCO2l_CO``, ``CRttbarCO2l_CO_2b``, ``CRttbarCO2l_gstr``,
``CRttbarCO2l_gstr_2b``, ``CR1b3lem``, ``CR1b3le``, ``CR1b3lm``,
``CRttW2l_plus``, ``CRttW2l_minus``, ``CR1bplus``, ``CR1bminus``,
``CRttW2l``, ``VRttZ3l``, ``VRttWCRSR``, ``SR4b``, ``SR2b``,
``SR3b``, ``SR2b2l``, ``SR2b3l4l``, ``SR2b4l``, ``SR3b2l``,
``SR3b3l4l``, ``SR3b4l``, ``SR4b4l``, ``SR``.
