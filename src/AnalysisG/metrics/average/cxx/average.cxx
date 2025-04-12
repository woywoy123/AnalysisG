#include <metrics/average.h>

average_metric::average_metric(){this -> name = "average";}
average_metric::~average_metric(){}
average_metric* average_metric::clone(){return new average_metric();}
