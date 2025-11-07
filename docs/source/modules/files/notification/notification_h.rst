notification.h
==============

**File Path**: ``modules/notification/include/notification/notification.h``

**File Type**: H (Header)

**Lines**: 51

Dependencies
------------

**Includes**:

- ``iostream``
- ``sstream``
- ``string``
- ``thread``
- ``vector``

Classes
-------

``notification``
~~~~~~~~~~~~~~~~

**Methods**:

- ``void success(std::string message)``
- ``void warning(std::string message)``
- ``void failure(std::string message)``
- ``void info(std::string message)``
- ``void progressbar(float prog, std::string title)``
- ``void progressbar(std::vector<size_t>* threads, std::vector<size_t>*...)``
- ``int running(std::vector<std::thread*>* thr, std::vector<size_t...)``
- ``void monitor(std::vector<std::thread*>* thr)``
- ``static progressbar1(std::vector<size_t>* threads, size_t l, std::strin...)``
- ``static progressbar2(std::vector<size_t>* threads, size_t* l, std::stri...)``

