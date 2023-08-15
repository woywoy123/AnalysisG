#include "metadata.h"

namespace SampleTracer
{
    CyMetaData::CyMetaData(){}
    CyMetaData::~CyMetaData(){}

    void CyMetaData::addsamples(int index, std::string sample)
    {
        this -> inputfiles[index] = sample; 
    }

    void CyMetaData::addconfig(std::string key, std::string val)
    {
        this -> config[key] = val; 
    }

}
