from AnalysisG.Generators import Analysis
from AnalysisG.Events.Events.Event import Event
from conftest import clean_dir

smpl = "./samples/"
Files = {
    smpl + "sample1": ["smpl1.root"],
    smpl + "sample2": ["smpl1.root", "smpl2.root", "smpl3.root"],
}

def test_analysis_event_merge():
    def EventGen_(Dir, Name):
        Ana = Analysis()
        Ana.ProjectName = "Project"
        Ana.InputSample(Name, Dir)
        Ana.EventCache = True
        Ana.Event = Event
        Ana.Threads = 10
        Ana.EventStart = 0
        Ana.EventStop = 10
        Ana.Launch()
        return Ana

    ev1 = EventGen_(File0, "Top")
    for i in ev1: assert i.Event
    ev2 = EventGen_(File1, "Tops")
    for i in ev2: assert i.Event
    ev1 += ev2
    it = 0
    for i in ev1:
        assert ev1[i.hash].hash == i.hash
        it += 1
    ev1.EventCache = False
    assert it == 20

    a_ev = EventGen_(None, None)
    assert len(ev1) == len(a_ev)
    clean_dir()

def test_Analysis():
    Sample1 = {smpl + "sample1": ["smpl1.root"]}
    Sample2 = smpl + "sample2"

    Ana = Analysis()
    Ana.ProjectName = "_Test"
    Ana.InputSample("Sample1", Sample1)
    Ana.InputSample("Sample2", Sample2)
    Ana.PurgeCache = False
    Ana.OutputDirectory = "../test/"
    Ana.EventStop = 100
    assert Ana.Launch() == False
    clean_dir()

def _template(default=True):
    AnaE = Analysis()
    AnaE.ProjectName = "Project"
    if default == True:
        AnaE.InputSample("Sample1", smpl + "sample1/" + Files[smpl + "sample1"][0])
        AnaE.InputSample("Sample2", smpl + "sample2/" + Files[smpl + "sample2"][1])
    else:
        AnaE.InputSample(**default)
    AnaE.Threads = 2
    AnaE.Verbose = 1
    return AnaE

def test_analysis_event_nocache():
    AnaE = _template()
    AnaE.Event = Event
    AnaE.Launch()
    assert len([i for i in AnaE]) != 0

def test_analysis_event_nocache_nolaunch():
    AnaE = _template()
    AnaE.Event = Event
    assert len([i for i in AnaE]) != 0

def test_analysis_event_cache():
    AnaE = _template()
    AnaE.Event = Event
    AnaE.EventCache = True
    AnaE.Launch()
    assert len([i for i in AnaE if i.Event]) != 0

    AnaE = _template()
    AnaE.EventCache = True
    AnaE.Verbose = 3
    AnaE.Launch()

    assert len([i for i in AnaE if i.Event]) != 0
    clean_dir()

def test_analysis_event_cache_diff_sample():
    Ana1 = _template(
        {
            "Name": "sample2",
            "SampleDirectory": smpl + "sample2/" + Files[smpl + "sample2"][1],
        }
    )
    Ana1.Event = Event
    Ana1.EventCache = True
    Ana1.Launch()

    assert len([i for i in Ana1]) != 0

    Ana2 = _template(
        {
            "Name": "sample1",
            "SampleDirectory": smpl + "sample1/" + Files[smpl + "sample1"][0],
        }
    )
    Ana2.Event = Event
    Ana2.EventCache = True
    Ana2.Launch()

    assert len([i for i in Ana2]) != 0

    AnaE = _template()
    AnaE.Event = Event
    AnaE.EventCache = True

    AnaS = Ana2 + Ana1
    assert len([i for i in AnaE if i.hash not in AnaS]) == 0
    clean_dir()


