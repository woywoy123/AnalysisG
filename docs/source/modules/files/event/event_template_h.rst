event_template.h
================

**File Path**: ``modules/event/include/templates/event_template.h``

**File Type**: H (Header)

**Lines**: 102

Dependencies
------------

**Includes**:

- ``meta/meta.h``
- ``structs/element.h``
- ``structs/event.h``
- ``structs/property.h``
- ``templates/particle_template.h``
- ``tools/tools.h``

Classes
-------

``event_template``
~~~~~~~~~~~~~~~~~~

**Inherits from**: ``tools``

**Methods**:

- ``static set_trees(std::vector<std::string>*, event_template*)``
- ``static set_branches(std::vector<std::string>*, event_template*)``
- ``static get_leaves(std::vector<std::string>*, event_template*)``
- ``void add_leaf(std::string key, std::string leaf = "")``
- ``static set_name(std::string*, event_template*)``
- ``static set_hash(std::string*, event_template*)``
- ``static get_hash(std::string*, event_template*)``
- ``static set_tree(std::string*, event_template*)``
- ``static get_tree(std::string*, event_template*)``
- ``static set_weight(double*, event_template*)``

