#include <tools/tools.h>
#include <sys/stat.h>
#include <filesystem>
#include <unistd.h>

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

std::vector<std::string> tools::ls(std::string path, std::string ext){
    if (this -> ends_with(&path, "*")){path = this -> split(path, "*")[0];}
    std::vector<std::string> out = {}; 
    std::filesystem::recursive_directory_iterator itr; 
    try {itr = std::filesystem::recursive_directory_iterator(path);}
    catch (...) {return {};}
    for (const std::filesystem::directory_entry val : itr){
        std::string s = std::filesystem::canonical(val.path()).string(); 
        if (ext.size() && !this -> ends_with(&s, ext)){continue;}
        out.push_back(s); 
    }
    return out; 
}

std::string tools::absolute_path(std::string path){
    return std::filesystem::canonical(path).string(); 
}
