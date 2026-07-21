#include <tools/tools.h>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>

void AnalysisG::tooling::create_path(std::string input_path){
    bool f = false;
    if (AnalysisG::tooling::split(input_path, ".").size() > 1){f = true;}

    std::vector<std::string> cuts = AnalysisG::tooling::split(input_path, "/"); 
    std::string path = ""; 
    for (unsigned int x(0); x < cuts.size() - f; ++x){
        path += cuts[x] + "/"; 
        mkdir(path.c_str(), S_IRWXU);
    }
}

void AnalysisG::tooling::rename(std::string start, std::string target){
    std::filesystem::rename(start, target); 
}

void AnalysisG::tooling::delete_path(std::string input_path){
    struct stat sb;
    if (!stat(input_path.c_str(), &sb)){
        if (S_ISDIR(sb.st_mode)){rmdir(input_path.c_str());}
        else {unlink(input_path.c_str());}
    }
}

bool AnalysisG::tooling::is_file(std::string path){
    return std::filesystem::is_regular_file(path); 
}

std::vector<std::string> AnalysisG::tooling::ls(std::string path, std::string ext){
    if (AnalysisG::tooling::ends_with(&path, "*")){path = AnalysisG::tooling::split(path, "*")[0];}
    std::vector<std::string> out = {}; 
    std::filesystem::recursive_directory_iterator itr; 
    try {itr = std::filesystem::recursive_directory_iterator(path);}
    catch (...) {return {};}
    for (const std::filesystem::directory_entry& val : itr){
        std::string s = ""; 
        try {s = std::filesystem::canonical(val.path()).string();}
        catch (...){continue;}
        if (!AnalysisG::tooling::is_file(s)){
            std::vector<std::string> vs = AnalysisG::tooling::ls(s + "*", ext); 
            for (size_t x(0); x < vs.size(); ++x){
                if (AnalysisG::tooling::is_file(vs[x])){out.push_back(vs[x]); continue;}
                std::vector<std::string> lx = AnalysisG::tooling::ls(vs[x] + "*"); 
                AnalysisG::tooling::unique_key(&lx, &out); 
            }
            continue;
        }
        if (ext.size() && !AnalysisG::tooling::ends_with(&s, ext)){continue;}
        out.push_back(s); 
    }
    std::vector<std::string> ox = {}; 
    AnalysisG::tooling::unique_key(&out, &ox); 
    return ox; 
}

std::string AnalysisG::tooling::absolute_path(std::string path){
    return std::filesystem::canonical(path).string(); 
}
