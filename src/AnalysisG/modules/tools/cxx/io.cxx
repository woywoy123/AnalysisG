#include <tools/merge_cast.h>
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

void tools::rename(std::string start, std::string target){
    std::filesystem::rename(start, target); 
}

void tools::delete_path(std::string input_path){
    struct stat sb;
    if (!stat(input_path.c_str(), &sb)){
        if (S_ISDIR(sb.st_mode)){rmdir(input_path.c_str());}
        else {unlink(input_path.c_str());}
    }
}

bool tools::is_file(std::string path){
    return std::filesystem::is_regular_file(path); 
}

std::vector<std::string> tools::ls(std::string path, std::string ext){
    if (tools::ends_with(&path, "*")){path = tools::split(path, "*")[0];}
    std::vector<std::string> out = {}; 
    std::vector<std::string> dix = {}; 
    std::filesystem::recursive_directory_iterator itr; 
    try {itr = std::filesystem::recursive_directory_iterator(path);}
    catch (...) {return {};}
    for (const std::filesystem::directory_entry& val : itr){
        std::string s = ""; 
        try {s = std::filesystem::canonical(val.path()).string();}
        catch (...){continue;}
        bool isF = tools::is_file(s); 
        if (!isF){dix.push_back(s + "*"); continue;}

        bool isE = tools::ends_with(&s, ext); 
        if (!isE){continue;}
        out.push_back(s);
    }
    std::map<std::string, bool> vx; 
    std::vector<std::string> ox = {}; 
    for (size_t y(0); y < dix.size(); ++y){
        if (vx[dix[y]]){continue;}
        std::vector<std::string> vs = tools::ls(dix[y], ext);
        for (size_t x(0); x < vs.size(); ++x){out.push_back(vs[x]);}
        vx[dix[y]] = true;
    }
    tools::unique_key(&out, &ox); 
    return ox; 
}

std::string tools::absolute_path(std::string path){
    return std::filesystem::canonical(path).string(); 
}

std::string tools::as_path(std::string base, std::string apn){
    base += std::string( tools::ends_with(&base, "/") ? "" : "/" ); 
    return base + apn; 
}

std::string tools::as_path(std::string base, bool cnd){
    if (!cnd){return base;}
    return base + std::string( ( tools::ends_with(&base, "/") ? "" : "/") ); 
}


