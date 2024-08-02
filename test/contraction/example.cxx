#include <iostream>
#include <vector>

class particle {

    public:
        particle(){}; 
        ~particle(){};

        int x = 1; 
        int y = 1; 
        double t = 1; 

}; 

int main(){
    std::cout << "...." << std::endl;
    size_t l = 100000000; 
    particle** p = new particle*[l];
    std::cout << "..>." << std::endl;
    for (size_t x(0); x < l; ++x){p[x] = new particle();}  
    std::cout << "...." << std::endl;
    for (size_t x(0); x < l; ++x){free(p[x]);} 
    free(p);

    std::cout << "...." << std::endl;

    std::vector<particle*>* tmp = new std::vector<particle*>(); 
    tmp -> reserve(l); 
    for (size_t x(0); x < l; ++x){tmp -> push_back(new particle());}
    for (size_t x(0); x < l; ++x){delete (*tmp)[x];}
    tmp -> clear(); 
    std::vector<particle*>().swap(*tmp); 
    std::cout << "+++" << std::endl; 
    for (int x(0); x < 1000; ++x){--x;} 
    return 0; 
} 
