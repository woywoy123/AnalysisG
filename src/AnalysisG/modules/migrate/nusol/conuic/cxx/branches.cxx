#include <conuic/branches.h>
#include <conuic/atomics.h>

branches_t::branches_t(){}

branches_t::branches_t(long double p_, long double m_, std::string name_){
    this -> p = p_; this -> m = m_; this -> name = name_; 
}

branches_t::branches_t(angular_t p_, angular_t m_, std::string name_){
    this -> pA = p_; this -> mA = m_; this -> name = name_; 
}

long double branches_t::pair(int sign){return signs(sign, this -> p ,  this -> m);}
angular_t   branches_t::Apair(int sign){return signs(sign, this -> pA, this -> mA);}

branches_t::~branches_t(){}


