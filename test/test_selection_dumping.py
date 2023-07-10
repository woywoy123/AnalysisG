from AnalysisG.Generators import EventGenerator, SelectionGenerator
from examples.ExampleSelection import Example, Example2
from examples.Event import EventEx
from AnalysisG import Analysis
from AnalysisG.IO import nTupler
#from conftest import clean_dir

smpl = "./samples/"
Files = {
    smpl + "sample1": ["smpl1.root"],
    smpl + "sample2": ["smpl1.root", "smpl2.root", "smpl3.root"],
}

def test_selection_generator():
    #Ev = EventGenerator(Files)
    #Ev.OutputDirectory = "Project"
    #Ev.Event = EventEx
    #Ev.Threads = 1
    #Ev.EventStop = 100
    #Ev.MakeEvents

    #sel = SelectionGenerator(Ev)
    #sel.OutputDirectory = "Project"
    #sel += Ev
    #sel.Threads = 2
    #sel.AddSelection("Example", Example)
    #sel.AddSelection("Example2", Example2)
    #sel.MergeSelection("Example2")
    #sel.MergeSelection("Example")
    #sel.MakeSelection
    n = nTupler("Project/Selections/Merged/")


    #res = UnpickleObject("./Project/Selections/Merged/Example2")
    #assert res.CutFlow["Success->Example"] == len(Ev)
    #assert len(res.TimeStats) == len(Ev)
    #assert len(Ev) * 4 == len(res.Top["Truth"])
    #clean_dir()

if __name__ == "__main__":
    test_selection_generator()

