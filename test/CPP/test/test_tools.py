def test_hash():
    from AnalysisG.Tools import Hash
    assert Hash("test") == "0xd1d16a4a0a7a19fb"

def test_pickle():
    from AnalysisG.IO import PickleObject, UnpickleObject

    x = {"..." : ["test"]}
    PickleObject(x, "Test")
    p = UnpickleObject("Test")
    assert x == p



if __name__ == "__main__":
    test_hash()
    test_pickle()
