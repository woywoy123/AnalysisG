from examples.ExampleSelection import Example, Example2
from AnalysisG.Generators import SelectionGenerator
from AnalysisG.Generators import EventGenerator
from examples.Event import EventEx
from AnalysisG.IO import nTupler

smpl = "./samples/"
Files = {
    smpl + "sample1": ["smpl1.root"],
    smpl + "sample2": ["smpl1.root", "smpl2.root", "smpl3.root"],
}
def template():
    EvtGen = EventGenerator(Files)
    EvtGen.EventStop = 100
    EvtGen.EventStart = 10
    EvtGen.Event = EventEx
    EvtGen.Threads = 1
    EvtGen.MakeEvents()
    return EvtGen

def test_selection_generator():
    ev = template()
    hsh = {}
    for i in ev: hsh[i.hash] = i

    sel = SelectionGenerator(ev)
    sel.Threads = 2
    sel.AddSelection(Example)
    sel.AddSelection(Example2)
    sel.MakeSelections()
    sel.GetEvent = False
    x = []
    assert "nominal/EventEx" in sel.ShowLength
    assert "nominal/Example" in sel.ShowLength
    assert "nominal/Example2" in sel.ShowLength
    for i in sel:
        assert i.SelectionName == "Example"
        assert "Truth" in i.Top
        assert i.hash in hsh
        e = hsh[i.hash]
        assert e.ROOT == i.ROOT
        assert e.index == i.index
        x.append(i)
    assert len(x) >= 90

def test_selection_ntupler():
    #ev = template()
    #sel = SelectionGenerator(ev)
    #sel.ProjectName = "Project"
    #sel.Threads = 2
    #sel.AddSelection(Example)
    #sel.AddSelection(Example2)
    #sel.MakeSelections()
    #sel.DumpSelections()
    #sel.DumpTracer()

    nt = nTupler()
    nt.ProjectName = "Project"
    nt.This("Example2 -> ", "nominal")
    nt.__start__()



if __name__ == "__main__":
    #test_selection_generator()
    test_selection_ntupler()
