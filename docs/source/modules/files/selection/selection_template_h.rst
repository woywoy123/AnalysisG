selection_template.h
====================

**File Path**: ``modules/selection/include/templates/selection_template.h``

**File Type**: H (Header)

**Lines**: 199

Dependencies
------------

**Includes**:

- ``meta/meta.h``
- ``structs/enums.h``
- ``structs/event.h``
- ``structs/property.h``
- ``templates/event_template.h``
- ``templates/particle_template.h``
- ``tools/merge_cast.h``
- ``tools/tools.h``
- ``tools/vector_cast.h``

Classes
-------

``selection_template``
~~~~~~~~~~~~~~~~~~~~~~

**Inherits from**: ``tools``

**Methods**:

- ``static set_name(std::string*, selection_template*)``
- ``static get_name(std::string*, selection_template*)``
- ``static set_hash(std::string*, selection_template*)``
- ``static get_hash(std::string*, selection_template*)``
- ``static get_tree(std::string*, selection_template*)``
- ``static set_weight(double*, selection_template*)``
- ``static get_weight(double*, selection_template*)``
- ``static set_index(long*, selection_template*)``
- ``bool selection(event_template* ev)``
- ``bool strategy(event_template* ev)``

