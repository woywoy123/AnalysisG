#ifndef EVENT_GENERATOR_H
#define EVENT_GENERATOR_H

#include <io/io.h>
#include <tools/tools.h>
#include <notification/notification.h>

class eventgenerator: 
    public tools,
    public notification
{
    public:
        eventgenerator();
        ~eventgenerator();

}; 

#endif
