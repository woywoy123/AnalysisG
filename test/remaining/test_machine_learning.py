from AnalysisG.Generators import RandomSamplers
from AnalysisG import Analysis
from AnalysisG.Events import Event, GraphChildren
from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Templates import FeatureAnalysis
from AnalysisG.Generators import Optimizer
from conftest import clean_dir

root1 = "./samples/sample1/smpl1.root"


def test_optimizer():
    from models.CheatModel import CheatModel

    Ana = Analysis()
    Ana.InputSample(None, root1)
    Ana.Event = Event
    Ana.ProjectName = "TestOptimizer"
    Ana.EventGraph = GraphChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.EventStop = 100
    Ana.kFolds = 4
    Ana.DataCache = True
    Ana.PurgeCache = True
    Ana.Launch()

    op = Optimizer(Ana)
    op.ProjectName = "TestOptimizer"
    op.Model = CheatModel
    op.Device = "cpu"
    op.Optimizer = "ADAM"
    op.OptimizerParams = {"lr": 0.001}
    op.ContinueTraining = False
    op.EnableReconstruction = True
    op.Batch = 1
    op.Epochs = 20
    op.Launch()

    clean_dir()


def test_optimizer_analysis():
    from models.CheatModel import CheatModel

    Ana = Analysis()
    Ana.InputSample(None, root1)
    Ana.ProjectName = "TestOptimizerAnalysis"
    Ana.Event = Event
    Ana.EventGraph = GraphChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.DataCache = True
    Ana.kFolds = 2
    Ana.kFold = 1
    Ana.Epochs = 20
    Ana.Optimizer = "ADAM"
    Ana.RunName = "RUN"
    Ana.DebugMode = True
    Ana.OptimizerParams = {"lr": 0.001}
    Ana.Scheduler = "ExponentialLR"
    Ana.ContinueTraining = False
    Ana.SchedulerParams = {"gamma": 1}
    Ana.Device = "cpu"
    Ana.Model = CheatModel
    Ana.EnableReconstruction = True
    Ana.BatchSize = 1
    Ana.Launch()
    clean_dir()


def test_parallel_analysis():
    from models.CheatModel import CheatModel

    Ana = Analysis()
    Ana.ProjectName = "Project"
    Ana.InputSample(None, root1)
    Ana.Event = Event
    Ana.EventGraph = GraphChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.DataCache = True
    Ana.Launch()

    Ana = Analysis()
    Ana.ProjectName = "Project"
    Ana.DataCache = True
    Ana.TrainingSize = 50
    Ana.kFolds = 10
    Ana.Launch()

    Ana = Analysis()
    Ana.ProjectName = "Project"
    Ana.Epochs = 5
    Ana.kFold = ["k-1", "k-2"]
    Ana.Optimizer = "ADAM"
    Ana.OptimizerParams = {"lr": 0.001}
    Ana.Device = "cpu"
    Ana.Model = CheatModel
    Ana.ContinueTraining = True
    Ana.BatchSize = 1
    Ana.Launch()

    Ana = Analysis()
    Ana.ProjectName = "Project"
    Ana.Epochs = 5
    Ana.kFold = ["k-1"]
    Ana.ContinueTraining = True
    Ana.Optimizer = "ADAM"
    Ana.OptimizerParams = {"lr": 0.001}
    Ana.Device = "cpu"
    Ana.Model = CheatModel
    Ana.BatchSize = 1
    Ana.Launch()

    clean_dir()


if __name__ == "__main__":
    test_random_sampling()
    test_feature_analysis()
    test_optimizer()
    test_optimizer_analysis()
    test_parallel_analysis()
    pass
