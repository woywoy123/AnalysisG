The EventTemplate Source Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The event class needs to be declared within the `event.h` header files, and might look something like the code below:

.. code:: C++

   #ifndef EVENTS_EVENTNAME_H
   #define EVENTS_EVENTNAME_H

   // header 
   #include <templates/event_template.h>
   #include "particles.h" // the definition of the particles to be used.

   class event_name: public event_template
   {
        public: 
            event_name(); 
            ~event_name() override; // overrides the destructor
       
            std::vector<custom_particle*> some_objects = {};  
            std::vector<particle_template*> some_particles = {};

            float some_variable = 0; 

            // is needed to make duplicate instances during the compilation process.
            event_template* clone() override; 

            // the method to build the event, more on element_t later
            void build(element_t* el) override; 

            // add any additional compilation steps to the event e.g. truth matching
            void CompileEvent() override; 

        private:
            std::map<std::string, custom_particle*> m_some_objects = {}; 
            std::map<std::string, custom_particle_v2*> m_some_particles = {};  
   }; 

   #endif

Now to the `event.cxx` file, in here the event is defined. 
But as one can see from the above, there are a few methods that are being marked as `override`, these simply the specify how the event is to be defined.
A brief explanation of these is given below:
- `clone`: A dummy method used to make clones of the event. 
- `build`: A method used to extract data from ROOT n-tuples per event and allows the user to assign attributes such as `some_variable` to the event.
- `CompileEvent`: An optional method used to build the event, this might include truth matching or something else.

Below is a simple example of what an event might look like

.. code:: C++

   #include "event.h"

   event_name::event_name(){
        this -> name = "event_name" // <-- give the event implementation some name 

        // tell the framework to fetch the leaf from the ROOT file.
        this -> add_leaf("some_variable", "some_very_long_variable_name_in_root"); 

        // tell the framework to fetch the tree that holds the leaf variable
        this -> trees = {"some-tree"}; 

        // register the event particles that the framework should fetch data for 
        this -> register_particles(&this -> m_some_objects); 
        this -> register_particles(&this -> m_some_particles); 
   }

   event_name::~event_name(){}

   event_template* event_name::clone(){return (event_template*)new event_name();}

   void event_name::build(element_t* el){
        // assign the variable a value based on leaf key
        el -> get("some_variable", &this -> some_variable);
   }

   void event_name::CompileEvent(){

        // do something with the particles that the framework has compiled
        std::map<std::string, custom_particle*>::iterator itr = this -> m_some_objects.begin(); 
        for (; itr != this -> m_some_objects.end(); ++itr){
            this -> some_objects.push_back(itr -> second); 
        }

        // downcast to parent particle template
        std::map<std::string, custom_particle_v2*>::iterator itr2 = this -> m_some_particles.begin(); 
        for (; itr2 != this -> m_some_particles.end(); ++itr2){
            this -> some_particles.push_back((particle_template*)itr -> second); 
        }
   }


From the above code, there is a few things that have not been explained yet, namely the usage of the `element_t` struct.
This struct holds the requested leaves, branches and tree data on an event by event basis, and uses the `get` function to automatically deduce the type that the key is mapped to.
In fact, a fair bit of magic occurs under the hood, but the main message is that the `get` function will cast the input type back to the user. 
For example, to request a `std::vector<std::vector<float>>` value from `element_t` is as simple as:

.. code:: C++

   // define the type
   std::vector<std::vector<float>> some_variable; 
    
   // use element_t (here called el, following from the above example)
   el -> get("some-varible", &some_variable); 

Also notice that the `get` call expects a different key than what is given by the ROOT n-tuples leaves.
This is an optional quirk if you are lazy and can't be bothered typing long names out.

The next part to point out is the private declarations of the particle maps. 
In the framework, particles are given a unique identifier in the form of a hash string. 
So when the particle registration occurs in the constructor of the event class, the framework knows to delete these particles once they are not needed.
This is discussed more in detail in the next section.

For more information about the methods and attributes of the event_template class, see the core-class documentation :ref:`event-template`.



