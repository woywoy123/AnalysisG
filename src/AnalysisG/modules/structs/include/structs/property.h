/**
 * @file property.h
 * @brief Defines the cproperty template class for property-based access patterns.
 *
 * This file contains the declaration of the `cproperty` template class, which provides
 * a property system similar to those found in languages like C# or Python. It allows
 * defining getter and setter functions that are automatically called when the property
 * is read or written, enabling validation, lazy evaluation, and computed properties.
 */

#ifndef PROPERTY_STRUCTS_H
#define PROPERTY_STRUCTS_H
#include <functional>
#include <string>

/**
 * @brief Default setter function that does nothing.
 * @tparam T The type of the property value.
 * @tparam G The type of the owning object.
 * @param val Pointer to the value (unused).
 * @param obj Pointer to the owning object (unused).
 */
template <typename T, typename G>
void x_setter(T*, G*){}

/**
 * @brief Default getter function that does nothing.
 * @tparam T The type of the property value.
 * @tparam G The type of the owning object.
 * @param val Pointer to the value (unused).
 * @param obj Pointer to the owning object (unused).
 */
template <typename T, typename G>
void x_getter(T*, G*){}

/**
 * @class cproperty
 * @brief A template class implementing a property system with getter/setter callbacks.
 *
 * The cproperty class provides a property-like interface where reading or writing
 * the property value can trigger custom getter or setter functions. This is useful for:
 * - Validating values before assignment
 * - Computing values on-demand (lazy evaluation)
 * - Synchronizing values with internal state
 * - Providing Python-friendly property access via Cython bindings
 *
 * @tparam T The type of the property value.
 * @tparam G The type of the owning object (used for callbacks).
 *
 * @section cproperty_usage Usage Example
 *
 * ```cpp
 * class MyClass {
 * public:
 *     cproperty<int, MyClass> value;
 *     
 *     MyClass() {
 *         value.set_getter(get_value);
 *         value.set_setter(set_value);
 *         value.set_object(this);
 *     }
 *     
 * private:
 *     int m_value = 0;
 *     
 *     static void get_value(int* out, MyClass* obj) {
 *         *out = obj->m_value;
 *     }
 *     
 *     static void set_value(int* in, MyClass* obj) {
 *         if (*in >= 0) obj->m_value = *in;  // Validation
 *     }
 * };
 *
 * // Usage:
 * MyClass obj;
 * obj.value = 42;      // Calls set_value
 * int x = obj.value;   // Calls get_value
 * ```
 */
template <typename T, typename G>
class cproperty 
{
    public: 
        /**
         * @brief Default constructor.
         * Initializes the property with a default-constructed value.
         */
        cproperty() : data() {}; 
        
        /**
         * @brief Sets the setter callback function.
         * @param c The setter function to call when the property is assigned.
         *          Signature: void(T* value, G* owner)
         */
        void set_setter(std::function<void(T*, G*)> c = x_setter<T, G>){
            this -> has_setter = true;
            this -> setter = c; 
        }

        /**
         * @brief Sets the getter callback function.
         * @param c The getter function to call when the property is read.
         *          Signature: void(T* value, G* owner)
         */
        void set_getter(std::function<void(T*, G*)> c = x_getter<T, G>){
            this -> has_getter = true; 
            this -> getter = c; 
        }

        /**
         * @brief Sets the owning object for callback invocation.
         * @param obj Pointer to the object that owns this property.
         */
        void set_object(G* obj){this -> obj = obj;}

        /**
         * @brief Assignment operator.
         * Assigns a value to the property and calls the setter if defined.
         * @param val The value to assign.
         * @return Reference to this property.
         */
        cproperty& operator=(const T& val){
            this -> data = val; 
            if (!this -> has_setter){return *this;}
            this -> setter(&this -> data, this -> obj); 
            return *this; 
        }

        /**
         * @brief Addition operator.
         * @param val The value to add.
         * @return The result of adding val to the property value.
         */
        T    operator+(const T& val){return this -> data + val;}
        
        /**
         * @brief Equality comparison operator.
         * @param val The value to compare against.
         * @return True if the property value equals val.
         */
        bool operator==(const T& val){return this -> data == val;}
        
        /**
         * @brief Inequality comparison operator.
         * @param val The value to compare against.
         * @return True if the property value does not equal val.
         */
        bool operator!=(const T& val){return this -> data != val;}

        /**
         * @brief Stream output operator.
         * @param out The output stream.
         * @return Reference to the output stream.
         */
        std::ostream& operator<<(std::ostream& out){
            out << this -> data; 
            return out;  
        }

        /**
         * @brief Implicit conversion operator.
         * Converts the property to its value type, calling the getter if defined.
         * @return The property value.
         */
        operator T(){
            if (!this -> has_getter){return this -> data;}
            this -> getter(&this -> data, this -> obj); 
            return this -> data;
        }
       
        /**
         * @brief Address-of operator.
         * Returns a pointer to the property value, calling the getter first if defined.
         * @return Pointer to the property value.
         */
        T* operator&(){
            if (!this -> has_getter){return &this -> data;}
            this -> getter(&this -> data, this -> obj); 
            return &this -> data;
        }

        /**
         * @brief Clears the property value to its default state.
         */
        void clear(){
            if (!this -> has_getter){return;}
            this -> data = T();
        }


    private: 
        T data;                                   ///< The stored property value.
        G* obj = nullptr;                         ///< Pointer to the owning object.
        bool has_getter = false;                  ///< Flag indicating if a getter is set.
        bool has_setter = false;                  ///< Flag indicating if a setter is set.
        std::function<void(T*, G*)> setter;       ///< The setter callback function.
        std::function<void(T*, G*)> getter;       ///< The getter callback function.

}; 

#endif
