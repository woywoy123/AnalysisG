====================================================================================================
Complete C++ Module Documentation with Dependency Tracing
====================================================================================================

This document provides **comprehensive documentation** for all 22 C++ modules in AnalysisG,
including complete dependency tracing, class hierarchies, and API references.

.. contents::
   :local:
   :depth: 3

Module Overview
====================================================================================================

AnalysisG contains 22 C++ modules organized into functional categories:

**Core Templates**

- ``event``: 1 headers, 2 sources, 1 classes, 0 structs
- ``particle``: 1 headers, 5 sources, 3 classes, 0 structs
- ``graph``: 1 headers, 3 sources, 5 classes, 1 structs
- ``selection``: 1 headers, 3 sources, 2 classes, 0 structs
- ``metric``: 1 headers, 5 sources, 3 classes, 2 structs
- ``model``: 1 headers, 4 sources, 6 classes, 4 structs

**Analysis Infrastructure**

- ``analysis``: 1 headers, 8 sources, 1 classes, 0 structs
- ``lossfx``: 1 headers, 4 sources, 1 classes, 0 structs
- ``optimizer``: 1 headers, 1 sources, 2 classes, 0 structs
- ``meta``: 1 headers, 1 sources, 2 classes, 0 structs

**Data Management**

- ``io``: 1 headers, 4 sources, 1 classes, 0 structs
- ``container``: 1 headers, 2 sources, 1 classes, 1 structs
- ``dataloader``: 1 headers, 3 sources, 3 classes, 1 structs
- ``sampletracer``: 1 headers, 1 sources, 1 classes, 0 structs

**Utilities**

- ``tools``: 1 headers, 3 sources, 1 classes, 0 structs
- ``typecasting``: 3 headers, 2 sources, 0 classes, 2 structs
- ``notification``: 1 headers, 1 sources, 1 classes, 0 structs
- ``structs``: 12 headers, 6 sources, 12 classes, 16 structs

**Visualization**

- ``plotting``: 1 headers, 1 sources, 1 classes, 0 structs
- ``roc``: 1 headers, 1 sources, 1 classes, 1 structs
- ``metrics``: 1 headers, 3 sources, 1 classes, 1 structs

**Physics**

- ``nusol``: 21 headers, 23 sources, 12 classes, 28 structs

Module Dependency Graph
====================================================================================================

This section shows the dependency relationships between modules.

analysis
--------

**Dependencies**: generators, io, structs, templates

**Location**: ``src/AnalysisG/modules/analysis/``

**Files**:

- Headers: 1
- Sources: 8

**Classes (1)**:

- ``analysis``
  - Inherits from: ``notification, 
    public tools``
  - Defined in: ``analysis/include/AnalysisG/analysis.h``

container
---------

**Dependencies**: generators, meta, templates, tools

**Location**: ``src/AnalysisG/modules/container/``

**Files**:

- Headers: 1
- Sources: 2

**Classes (1)**:

- ``container``
  - Inherits from: ``tools``
  - Defined in: ``container/include/container/container.h``

**Structs (1)**:

- ``entry_t``

dataloader
----------

**Dependencies**: notification, structs, templates, tools

**Location**: ``src/AnalysisG/modules/dataloader/``

**Files**:

- Headers: 1
- Sources: 3

**Classes (3)**:

- ``analysis``
  - Defined in: ``dataloader/include/generators/dataloader.h``
- ``dataloader``
  - Inherits from: ``notification, 
    public tools``
  - Defined in: ``dataloader/include/generators/dataloader.h``
- ``model_template``
  - Defined in: ``dataloader/include/generators/dataloader.h``

**Structs (1)**:

- ``model_report``

event
-----

**Dependencies**: meta, structs, templates, tools

**Location**: ``src/AnalysisG/modules/event/``

**Files**:

- Headers: 1
- Sources: 2

**Classes (1)**:

- ``event_template``
  - Inherits from: ``tools``
  - Defined in: ``event/include/templates/event_template.h``

graph
-----

**Dependencies**: structs, templates, tools

**Location**: ``src/AnalysisG/modules/graph/``

**Files**:

- Headers: 1
- Sources: 3

**Classes (5)**:

- ``analysis``
  - Defined in: ``graph/include/templates/graph_template.h``
- ``container``
  - Defined in: ``graph/include/templates/graph_template.h``
- ``dataloader``
  - Defined in: ``graph/include/templates/graph_template.h``
- ``graph_template``
  - Defined in: ``graph/include/templates/graph_template.h``
- ``meta``
  - Defined in: ``graph/include/templates/graph_template.h``

**Structs (1)**:

- ``graph_t``

io
--

**Dependencies**: meta, notification, structs, tools

**Location**: ``src/AnalysisG/modules/io/``

**Files**:

- Headers: 1
- Sources: 4

**Classes (1)**:

- ``io``
  - Inherits from: ``tools, 
    public notification``
  - Defined in: ``io/include/io/io.h``

lossfx
------

**Dependencies**: notification, structs, tools

**Location**: ``src/AnalysisG/modules/lossfx/``

**Files**:

- Headers: 1
- Sources: 4

**Classes (1)**:

- ``lossfx``
  - Inherits from: ``tools, 
    public notification``
  - Defined in: ``lossfx/include/templates/lossfx.h``

meta
----

**Dependencies**: notification, structs, tools

**Location**: ``src/AnalysisG/modules/meta/``

**Files**:

- Headers: 1
- Sources: 1

**Classes (2)**:

- ``analysis``
  - Defined in: ``meta/include/meta/meta.h``
- ``meta``
  - Inherits from: ``tools,
    public notification``
  - Defined in: ``meta/include/meta/meta.h``

metric
------

**Dependencies**: meta, notification, plotting, structs, templates, tools

**Location**: ``src/AnalysisG/modules/metric/``

**Files**:

- Headers: 1
- Sources: 5

**Classes (3)**:

- ``analysis``
  - Defined in: ``metric/include/templates/metric_template.h``
- ``metric_template``
  - Defined in: ``metric/include/templates/metric_template.h``
- ``model_template``
  - Defined in: ``metric/include/templates/metric_template.h``

**Structs (2)**:

- ``graph_t``
- ``metric_t``

metrics
-------

**Dependencies**: notification, structs, templates

**Location**: ``src/AnalysisG/modules/metrics/``

**Files**:

- Headers: 1
- Sources: 3

**Classes (1)**:

- ``metrics``
  - Inherits from: ``tools, 
    public notification``
  - Defined in: ``metrics/include/metrics/metrics.h``

**Structs (1)**:

- ``analytics_t``

model
-----

**Dependencies**: notification, structs, templates

**Location**: ``src/AnalysisG/modules/model/``

**Files**:

- Headers: 1
- Sources: 4

**Classes (6)**:

- ``analysis``
  - Defined in: ``model/include/templates/model_template.h``
- ``dataloader``
  - Defined in: ``model/include/templates/model_template.h``
- ``metric_template``
  - Defined in: ``model/include/templates/model_template.h``
- ``metrics``
  - Defined in: ``model/include/templates/model_template.h``
- ``model_template``
  - Defined in: ``model/include/templates/model_template.h``
- ``optimizer``
  - Defined in: ``model/include/templates/model_template.h``

**Structs (4)**:

- ``graph_t``
- ``model_report``
- ``optimizer_params_t``
- ``variable_t``

notification
------------

**Location**: ``src/AnalysisG/modules/notification/``

**Files**:

- Headers: 1
- Sources: 1

**Classes (1)**:

- ``notification``
  - Defined in: ``notification/include/notification/notification.h``

nusol
-----

**Dependencies**: notification, reconstruction, structs, templates, tools

**Location**: ``src/AnalysisG/modules/nusol/``

**Files**:

- Headers: 21
- Sources: 23

**Classes (12)**:

- ``conics``
  - Defined in: ``nusol/tmp/conuix/include/conuix/conuix.h``
- ``conuic``
  - Defined in: ``nusol/conuix/include/conuix/conuix.h``
- ``conuix``
  - Defined in: ``nusol/nusol/include/reconstruction/nusol.h``
- ``ellipse``
  - Defined in: ``nusol/nusol/include/reconstruction/nusol.h``
- ``mtx``
  - Defined in: ``nusol/ellipse/include/ellipse/nusol.h``
- ``multisol``
  - Defined in: ``nusol/tmp/multisol.h``
- ``nuclx``
  - Defined in: ``nusol/tmp/conuix/include/conuix/nusol.h``
- ``nuelx``
  - Defined in: ``nusol/ellipse/include/ellipse/nusol.h``
- ``nusol``
  - Defined in: ``nusol/nusol/include/reconstruction/nusol.h``
- ``nusol_enum``
  - Defined in: ``nusol/nusol/include/reconstruction/nusol.h``
  - ... and 2 more

**Structs (28)**:

- ``H_matrix_t``
- ``P_t``
- ``Sx_t``
- ``Sy_t``
- ``atomics_t``
- ``base_t``
- ``dPdtau_t``
- ``debug``
- ``eig_t``
- ``ellipse_t``
  - ... and 18 more

optimizer
---------

**Dependencies**: generators, metrics, structs, templates

**Location**: ``src/AnalysisG/modules/optimizer/``

**Files**:

- Headers: 1
- Sources: 1

**Classes (2)**:

- ``analysis``
  - Defined in: ``optimizer/include/generators/optimizer.h``
- ``optimizer``
  - Inherits from: ``tools,
    public notification``
  - Defined in: ``optimizer/include/generators/optimizer.h``

particle
--------

**Dependencies**: structs, tools

**Location**: ``src/AnalysisG/modules/particle/``

**Files**:

- Headers: 1
- Sources: 5

**Classes (3)**:

- ``event_template``
  - Defined in: ``particle/include/templates/particle_template.h``
- ``particle_template``
  - Inherits from: ``tools``
  - Defined in: ``particle/include/templates/particle_template.h``
- ``selection_template``
  - Defined in: ``particle/include/templates/particle_template.h``

plotting
--------

**Dependencies**: notification, structs, tools

**Location**: ``src/AnalysisG/modules/plotting/``

**Files**:

- Headers: 1
- Sources: 1

**Classes (1)**:

- ``plotting``
  - Inherits from: ``tools, 
    public notification``
  - Defined in: ``plotting/include/plotting/plotting.h``

roc
---

**Dependencies**: plotting

**Location**: ``src/AnalysisG/modules/roc/``

**Files**:

- Headers: 1
- Sources: 1

**Classes (1)**:

- ``roc``
  - Inherits from: ``plotting``
  - Defined in: ``roc/include/plotting/roc.h``

**Structs (1)**:

- ``roc_t``

sampletracer
------------

**Dependencies**: container, notification

**Location**: ``src/AnalysisG/modules/sampletracer/``

**Files**:

- Headers: 1
- Sources: 1

**Classes (1)**:

- ``sampletracer``
  - Inherits from: ``tools, 
    public notification``
  - Defined in: ``sampletracer/include/generators/sampletracer.h``

selection
---------

**Dependencies**: meta, structs, templates, tools

**Location**: ``src/AnalysisG/modules/selection/``

**Files**:

- Headers: 1
- Sources: 3

**Classes (2)**:

- ``container``
  - Defined in: ``selection/include/templates/selection_template.h``
- ``selection_template``
  - Inherits from: ``tools``
  - Defined in: ``selection/include/templates/selection_template.h``

structs
-------

**Dependencies**: structs, tools

**Location**: ``src/AnalysisG/modules/structs/``

**Files**:

- Headers: 12
- Sources: 6

**Classes (12)**:

- ``cproperty``
  - Defined in: ``structs/include/structs/property.h``
- ``data_enum``
  - Defined in: ``structs/include/structs/enums.h``
- ``graph_enum``
  - Defined in: ``structs/include/structs/enums.h``
- ``loss_enum``
  - Defined in: ``structs/include/structs/enums.h``
- ``metrics``
  - Defined in: ``structs/include/structs/report.h``
- ``mlp_init``
  - Defined in: ``structs/include/structs/enums.h``
- ``mode_enum``
  - Defined in: ``structs/include/structs/enums.h``
- ``opt_enum``
  - Defined in: ``structs/include/structs/enums.h``
- ``optimizer_params_t``
  - Defined in: ``structs/include/structs/optimizer.h``
- ``particle_enum``
  - Defined in: ``structs/include/structs/enums.h``
  - ... and 2 more

**Structs (16)**:

- ``bsc_t``
- ``data_t``
- ``element_t``
- ``event_t``
- ``folds_t``
- ``graph_hdf5``
- ``graph_hdf5_w``
- ``loss_opt``
- ``meta_t``
- ``model_report``
  - ... and 6 more

tools
-----

**Location**: ``src/AnalysisG/modules/tools/``

**Files**:

- Headers: 1
- Sources: 3

**Classes (1)**:

- ``tools``
  - Defined in: ``tools/include/tools/tools.h``

typecasting
-----------

**Dependencies**: structs

**Location**: ``src/AnalysisG/modules/typecasting/``

**Files**:

- Headers: 3
- Sources: 2

**Structs (2)**:

- ``variable_t``
- ``write_t``
