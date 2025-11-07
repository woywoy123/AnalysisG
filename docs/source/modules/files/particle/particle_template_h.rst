particle_template.h
===================

**File Path**: ``modules/particle/include/templates/particle_template.h``

**File Type**: H (Header)

**Lines**: 166

Dependencies
------------

**Includes**:

- ``cmath``
- ``cstdlib``
- ``iostream``
- ``sstream``
- ``string``
- ``structs/element.h``
- ``structs/particles.h``
- ``structs/property.h``
- ``tools/tools.h``

Classes
-------

``particle_template``
~~~~~~~~~~~~~~~~~~~~~

**Inherits from**: ``tools``

**Methods**:

- ``explicit particle_template(particle_t* p)``
- ``explicit particle_template(particle_template* p, bool dump = false)``
- ``explicit particle_template(double px, double py, double pz, double e)``
- ``explicit particle_template(double px, double py, double pz)``
- ``static set_e(double*, particle_template*)``
- ``static get_e(double*, particle_template*)``
- ``static set_pt(double*, particle_template*)``
- ``static get_pt(double*, particle_template*)``
- ``static set_eta(double*, particle_template*)``
- ``static get_eta(double*, particle_template*)``

