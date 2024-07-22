The `cproperty` Type
--------------------

This type is declared within the struct module, and is used to trigger functions whenever an attribute has been assigned.
It is similar to using the `@property` and `@<function>.setter` methods in python.
For example, when the `name` of variable is assigned some variable, the `set_name` function is automatically called and modifies the `event_t` struct attributes.

.. cpp:class:: cproperty

   .. cpp:function:: cproperty()

   .. cpp:function:: void set_setter(std::function<void(T*, G*)> c) 

      Specifies the setter function.

   .. cpp:function:: void set_getter(std::function<void(T*, G*)> c)

      Specifies the getter function.

   .. cpp:function:: void set_object(G* obj)

      Sets the object that the property should interact with.

   .. cpp:function:: cproperty& operator=(const T& val)

   .. cpp:function:: bool operator==(const T& val)

   .. cpp:function:: bool operator!=(const T& val)

   .. cpp:function:: operator T()

   .. cpp:function:: void clear()

   .. cpp:function:: T* operator&()

Example Usage
^^^^^^^^^^^^^

.. code:: C++

   class some_object {
        public:
            some_object(){
                this -> secret.set_setter(this -> set_secret); 
                this -> secret.set_getter(this -> get_secret); 
                this -> secret.set_object(this);
            }
            ~some_object(){}


            // instance initialization
            cproperty<std::string, some_object*> secret; 


        private:
            void static get_secret(std::string* val, some_object* obj){
                *val = obj -> secret_value; 
            }

            void static set_secret(std::string* val, some_object* obj){
                obj -> secret_value = (*val); 
            }

            std::string secret_value = ""; 
    }; 

    some_object* ob = new some_object(); 
    ob -> secret = "super-cool";  // calls the set_secret function
    std::string sec = obj -> secret; // calls the get_secret function
