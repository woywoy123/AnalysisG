import os

def unit_memory(inpt):
    if len(inpt.Memory) < 3: return ""
    unit = inpt.Memory[-2:].lower()
    std  = inpt.Memory[:-2]
    try: tmp = float(std)
    except: return ""
    if unit == "gb": tmp *= 1024
    return "Request_Memory = " + str(tmp) + "\n"

def unit_time(inpt):
    if len(inpt.Time) < 2: return ""
    unit = inpt.Time.lower()
    if unit.endswith("h"): std = inpt.Time[:-1]
    elif unit.endswith("hrs"): std = inpt.Time[:-3]
    elif unit.endswith("m"): std = inpt.Time[:-1]
    elif unit.endswith("min"): std = inpt.Time[:-3]
    elif unit.endswith("mins"): std = inpt.Time[:-4]
    else: return ""

    unit = inpt.Time[len(std):]
    try: tmp = float(std)
    except: return ""

    if unit.startswith("h"): tmp *= 60*60
    elif unit.startswith("m"): tmp *= 60
    return "+RequestRuntime = " + str(tmp) + "\n"

def hardware(inpt):
    std = ""
    if "cuda" not in inpt.Device: pass
    else: std += "Request_GPUs = 1 \n"
    if not inpt.Threads: pass
    else: std += "Request_Cpus = " + inpt.Threads + "\n"
    return std


def build_shell_script(inpt):
    con = "#!/bin/bash\n"
    con += "source ~/.bashrc\n"
    if inpt.CondaVenv:
        con += 'eval "$(conda shell.bash hook)"' + "\n"
        con += "conda activate " + inpt.CondaVenv
    elif inpt.PythonVenv:
        if inpt.PythonVenv.startswith("$"): con += "source " + inpt.PythonVenv
        elif inpt.PythonVenv.endswith(".sh"): con += "source " + inpt.PythonVenv
        else:
            con += "alias " + inpt.PythonVenv + "\n"
            con += inpt.PythonVenv
    con += "\n"

    for jb in inpt.Jobs.values():
        jb.bash_script = con + "python "
        jb.bash_script += os.path.abspath(inpt.OutputDirectory) + "/"
        jb.bash_script += inpt.ProjectName + "/"
        jb.bash_script += "Condor/python_scripts/" + jb.JobName + ".py"

def build_condor_script(inpt):
    for jb in inpt.Jobs.values():
        exec_ = inpt.OutputDirectory + "/"
        exec_ += inpt.ProjectName + "/Condor"
        jb.condor_script  = "executable = " + exec_ + "/shells/" + jb.JobName + ".sh\n"
        jb.condor_script += "error = " + exec_ + "/condor/" + jb.JobName + ".error.$(ClusterID)\n"
        std = ""
        if not jb.OpSysVer: pass
        else: std = "Requirements = OpSysAndVer == " + jb.OpSysVer + "\n"

        jb.condor_script += std
        jb.condor_script += unit_memory(jb)
        jb.condor_script += unit_time(jb)
        jb.condor_script += hardware(jb)
        jb.condor_script += "queue 1"


def build_analysis_script(inpt):
    exec_ = inpt.OutputDirectory + "/"
    exec_ += inpt.ProjectName + "/"

    for jb in inpt.Jobs.values():
        jb.set_path = os.path.abspath(exec_ + "Condor/settings/" + jb.JobName)
        jb.py_path = os.path.abspath(exec_ + "Condor/python_scripts/")
        jb.python_script = "from AnalysisG import Analysis \n"
        jb.python_script += "from AnalysisG.IO import UnpickleObject\n"
        jb.python_script += "\n"
        jb.python_script += 'pkl = UnpickleObject("' + jb.set_path + '")\n'
        jb.python_script += "\n"
        jb.python_script += "ana = Analysis()\n"
        jb.python_script += "ana.ImportSettings(pkl)\n"
        jb.python_script += "ana.Launch()"

def build_dag(inpt):
    pth  = inpt.OutputDirectory + "/"
    pth += inpt.ProjectName + "/"
    pth += "Condor/"
    pth = os.path.abspath(pth) + "/"
    inpt._bash_path = [pth + "shells/main.sh"]
    inpt._dag_path  = pth + "condor/DAG_Submission.submit"

    jobs = []
    for jb in inpt.Jobs:
        x = "JOB " + jb + " "
        x += pth + "condor/" + jb + ".submit"
        jobs.append(x)
    inpt._dag_script = "\n".join(jobs + [""])

    out = []
    parnt = {i : list(inpt.Jobs[i].parents) for i in list(inpt.Jobs)}
    jn = [i for i in parnt if not len(parnt[i])][0]
    while True:
        if jn not in parnt: out += [jn]
        elif not len(parnt[jn]): out, _ = out + [jn], parnt.pop(jn)

        if not len(parnt): break
        try: jn = next(iter([i for i in parnt[jn] if i not in out]))
        except StopIteration: parnt[jn] = []
        except KeyError: jn = list(parnt).pop()


    x = []
    inpt._bash_path += ["#!/bin/bash\n\n"]
    for name in out:
        inpt._bash_path[1] += "bash " + pth + "shells/" + name + ".sh\n"
        x += ["PARENT " + i + " CHILD " + name for i in inpt.Jobs[name].parents]
    inpt._dag_script += "\n".join(list(set(x)))

