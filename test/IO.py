from AnalysisTopGNN.Tools import IO


def TestDirectory(Files):
    import os 
    from glob import glob

    def ListFilesInDir(directory, extension = None):
        srch = glob(directory + "/*") if extension == None else glob(directory + "/*" + extension)
        return [i for i in srch]
    
    def IsFile(directory):
        if os.path.isfile(directory):
            return directory
        else:
            return False

    def IOTest(directory, extension, _it = 0):
        F = []
        if isinstance(directory, dict):
            for i in directory:
                if isinstance(directory[i], list):
                    directory[i] = [i + "/" + k for k in directory[i]]
                else:
                    directory[i] = [i + "/" + directory[i]]
                F += IOTest([k for k in IOTest(directory[i], extension, _it+1)], extension, _it+1)
        elif isinstance(directory, list):
            F += [t for k in directory for t in IOTest(k, extension, _it+1)]
        elif isinstance(directory, str):
            if directory.endswith("*"):
                F += ListFilesInDir(directory[:-2], extension)
            else:
                F += [directory.replace("//", "/")]
            F = [i for i in F if IsFile(i)] 

        if _it == 0:
            dirs = {os.path.dirname(i) : [] for i in F}
            F = {i : [k.split("/")[-1] for k in F if os.path.dirname(k) == i] for i in dirs} 
        return F
        

 
    D = {Files + "/tttt" : ["*"], Files + "/t" : "QU_14.root", Files + "/ttbar/" : "*"}
    F = IO(D, ".root")
    
    return True

