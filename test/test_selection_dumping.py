from AnalysisG.Generators import EventGenerator, SelectionGenerator
from examples.ExampleSelection import Example, Example2
from examples.Event import EventEx
from AnalysisG import Analysis
from AnalysisG.IO import nTupler, UpROOT
from conftest import clean_dir
from time import sleep

smpl = "./samples/"
Files = {
    smpl + "sample1": ["smpl1.root"],
    smpl + "sample2": ["smpl1.root", "smpl2.root", "smpl3.root"],
}

def test_selection_generator():
    Ev = EventGenerator(Files)
    Ev.OutputDirectory = "Project"
    Ev.Event = EventEx
    Ev.Threads = 1
    Ev.EventStop = 100
    Ev.MakeEvents

    sel = SelectionGenerator(Ev)
    sel.OutputDirectory = "Project"
    sel += Ev
    sel.Threads = 2
    sel.AddSelection("Example", Example)
    sel.AddSelection("Example2", Example2)
    sel.MergeSelection("Example2")
    sel.MergeSelection("Example")
    sel.MakeSelection

    n = nTupler("Project/Selections/Merged/")
    n.This("Example2 -> Top", "nominal")
    n.This("Example -> Top", "nominal")
    tst = []
    for i in n: tst.append(i.Top); sleep(0.01)
    assert len(tst) == 200
    n.This("Example2 -> CutFlow", "nominal")
    n.Write()

    clean_dir()

if __name__ == "__main__":
    test_selection_generator()

