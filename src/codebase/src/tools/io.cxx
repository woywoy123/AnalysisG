#include <sys/stat.h>
#include <unistd.h>
#include <tools.h>

void tools::create_path(std::string input_path){
    bool f = false;
    if (tools::split(input_path, ".").size() > 1){f = true;}

    std::vector<std::string> cuts = tools::split(input_path, "/"); 
    std::string path = ""; 
    for (unsigned int x(0); x < cuts.size() - f; ++x){
        path += cuts[x] + "/"; 
        mkdir(path.c_str(), S_IRWXU);
    }
}

void tools::delete_path(std::string input_path){
    struct stat sb;
    if (!stat(input_path.c_str(), &sb)){
        if (S_ISDIR(sb.st_mode)){rmdir(input_path.c_str());}
        else {unlink(input_path.c_str());}
    }
}

bool tools::is_file(std::string path){
    struct stat buffer; 
    return (stat (path.c_str(), &buffer) == 0);
}

