#include "<selection-name>.h"

<selection-name>::<selection-name>(){this -> name = "<selection-name>";}
<selection-name>::~<selection-name>(){}

selection_template* <selection-name>::clone(){
    return (selection_template*)new <selection-name>();
}

void <selection-name>::merge(selection_template* sl){
    <selection-name>* slt = (<selection-name>*)sl; 

    // example variable
    merge_data(&this -> <var-name>, &slt -> <var-name>); 
}

bool <selection-name>::selection(event_template* ev){return true;}

bool <selection-name>::strategy(event_template* ev){
    <event-name>* evn = (<event-name>*)ev; 

    return true; 
}

