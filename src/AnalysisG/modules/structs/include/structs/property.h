#ifndef PROPERTY_STRUCTS_H
#define PROPERTY_STRUCTS_H
#include <functional>
#include <string>

template <typename T, typename G>
void x_setter(T*, G*){}

template <typename T, typename G>
void x_getter(T*, G*){}

template <typename T, typename G>
class cproperty 
{
    public: 
        cproperty() : data() {}; 
        void set_setter(std::function<void(T*, G*)> c = x_setter<T, G>){
            this -> has_setter = true;
            this -> setter = c; 
        }

        void set_getter(std::function<void(T*, G*)> c = x_getter<T, G>){
            this -> has_getter = true; 
            this -> getter = c; 
        }

        void set_object(G* obj){this -> obj = obj;}

        cproperty& operator=(const T& val){
            this -> data = val; 
            if (!this -> has_setter){return *this;}
            this -> setter(&this -> data, this -> obj); 
            return *this; 
        }

        T    operator+(const T& val){return this -> data + val;}
        bool operator==(const T& val){return this -> data == val;}
        bool operator!=(const T& val){return this -> data != val;}

        operator T(){
            if (!this -> has_getter){return this -> data;}
            this -> getter(&this -> data, this -> obj); 
            return this -> data;
        }
       
        T* operator&(){
            if (!this -> has_getter){return &this -> data;}
            this -> getter(&this -> data, this -> obj); 
            return &this -> data;
        }

        void clear(){
            if (!this -> has_getter){return;}
            this -> data = T();
        }


    private: 
        T data; 
        G* obj = nullptr; 
        bool has_getter = false; 
        bool has_setter = false; 
        std::function<void(T*, G*)> setter; 
        std::function<void(T*, G*)> getter; 

}; 

#endif
