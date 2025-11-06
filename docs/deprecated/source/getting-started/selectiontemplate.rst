The SelectionTemplate Source Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C++ Example Interface 
^^^^^^^^^^^^^^^^^^^^^

.. code:: C++ 

    #ifndef example_selection_H
    #define example_selection_H
    
    #include <<event-name>/event.h>
    #include <templates/selection_template.h>
    
    class example_selection: public selection_template
    {
        public:
            example_selection();
            ~example_selection() override; 
            selection_template* clone() override; 
    
            bool selection(event_template* ev) override; 
            bool strategy(event_template* ev) override;
            void merge(selection_template* sl) override;
    
    
            std::vector<float> <var-name>; 
    };

    #endif


.. code:: C++

    #include "<selection-name>.h"

    example_selection::example_selection(){this -> name = "example_selection";}
    example_selection::~example_selection(){}
    
    selection_template* example_selection::clone(){
        return (selection_template*)new example_selection();
    }
    
    void example_selection::merge(selection_template* sl){
        example_selection* slt = (example_selection*)sl; 
    
        // example variable
        this -> merge_data(&this -> <var-name>, &slt -> <var-name>); 
        this -> write(&this -> <var-name>, "some-name-for-ROOT"); 
    }
    
    bool example_selection::selection(event_template* ev){return true;}
    
    bool example_selection::strategy(event_template* ev){
        <event-name>* evn = (<event-name>*)ev; 
    
        return true; 
    }
    
   

Interfacing C++ code with Cython
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: Python

   # distuils: language=c++
   # cython: language_level=3
   # example_selection.pxd
   
   from libcpp.map cimport map
   from libcpp.vector cimport vector
   from libcpp.string cimport string
   from AnalysisG.core.selection_template cimport *
   
   cdef extern from "example_selection.h":
       cdef cppclass example_selection(selection_template):
           example_selection() except +
   
   cdef class ExampleSelection(SelectionTemplate):
       cdef example_selection* tt



.. code:: Python

   # distutils: language=c++
   # cython: language_level=3
   # example_selection.pyx
   
   from AnalysisG.core.tools cimport as_dict, as_list
   from AnalysisG.core.selection_template cimport *
   
   cdef class ExampleSelection(SelectionTemplate):
       def __cinit__(self):
           self.ptr = new example_selection()
           self.tt = <example_selection*>self.ptr
   
       def __dealloc__(self): del self.tt
   
       cdef void transform_dict_keys(self):
           #convert map keys to python string
           pass
   

 
