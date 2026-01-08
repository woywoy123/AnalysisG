The `cproperty` Type
--------------------

The `cproperty` type, found in the `struct` module, allows you to automatically call functions when an object's attribute is accessed or modified. This is similar to Python's `@property` decorator and its associated setter method.

For instance, if you assign a value to an attribute managed by `cproperty`, a predefined "setter" function is triggered. Similarly, when you read the attribute's value, a "getter" function is called.

Example Usage
^^^^^^^^^^^^^

.. code:: C++

    #include <string>
    #include <functional> // Required for std::function

    // Assuming cproperty is defined elsewhere, e.g., in "struct_module.h"
    // template<typename T, typename Owner> class cproperty { ... };

    class some_object {
    public:
         // Define the type for setter and getter functions
         using getter_func_t = std::function<void(std::string*, some_object*)>;
         using setter_func_t = std::function<void(std::string*, some_object*)>;

         some_object() {
              // Bind the static member functions to std::function objects
              getter_func_t getter = get_secret;
              setter_func_t setter = set_secret;

              // Configure the cproperty instance
              this->secret.set_getter(getter);
              this->secret.set_setter(setter);
              this->secret.set_object(this); // Pass the owner object instance
         }
         ~some_object() {}

         // Declare the cproperty member
         cproperty<std::string, some_object*> secret;

    private:
         // Static getter function
         static void get_secret(std::string* val, some_object* obj) {
              *val = obj->secret_value;
              // std::cout << "Getter called!" << std::endl; // Optional: for debugging
         }

         // Static setter function
         static void set_secret(std::string* val, some_object* obj) {
              obj->secret_value = *val;
              // std::cout << "Setter called with value: " << *val << std::endl; // Optional: for debugging
         }

         // The actual storage for the value
         std::string secret_value = "";
    };

    // --- Usage Example ---
    // #include <iostream> // Include if using std::cout for debugging

    int main() {
         some_object* ob = new some_object();

         // Assignment triggers the set_secret function
         ob->secret = "super-cool";

         // Reading the value triggers the get_secret function
         std::string sec = ob->secret;

         // std::cout << "Retrieved value: " << sec << std::endl; // Optional: verify value

         delete ob; // Clean up memory
         return 0;
    }
