from AnalysisG.Tools import Code
from AnalysisG.Tools import Tools, PickleObject, UnpickleObject
directory = Tools().abs("./samples/test/") + "/"

class simpleclass:
    def __init__(self, hello, world="world", test = None):
        pass

    def Test(self):
        return True


class simple_class_with_param:
    def __init__(self, hello = None):

        self.__params__ = {"key" : None, "key1" : None}

    def Test(self):
        return self.__params__


def f_default(test, x = "test"):
    return x


def f_nodefault(test, x):
    return (test, x)


def test_code_extraction():

    # test non initialized
    x = simpleclass

    c = Code(x)
    assert c.is_class
    assert not c.is_function
    assert c.is_callable
    assert not c.is_initialized
    assert not len(c.function_name)
    assert c.class_name == "simpleclass"
    assert "simpleclass" in c.source_code and "directory = " in c.source_code
    assert "simpleclass" in c.object_code and not "directory" in c.object_code
    assert None in c.defaults and "world" in c.defaults
    assert sum([True for i in ["hello", "world", "test"] if i in c.co_vars])
    assert {"hello": None, "world" : "world" , "test" : None} == c.input_params


    # test with initialized
    x = x("test")
    c2 = Code(x)
    assert c2.is_class
    assert not c2.is_function
    assert not c2.is_callable
    assert c2.is_initialized
    assert not len(c2.function_name)
    assert c2.class_name == "simpleclass"
    assert "simpleclass" in c2.source_code and "directory = " in c2.source_code
    assert "simpleclass" in c2.object_code and not "directory" in c2.object_code
    assert None in c2.defaults and "world" in c.defaults
    assert sum([True for i in ["hello", "world", "test"] if i in c2.co_vars])
    assert {"hello": None, "world" : "world" , "test" : None} == c2.input_params


    # test with defaults
    x = f_default

    c = Code(x)
    assert not c.is_class
    assert c.is_function
    assert c.is_callable
    assert not c.is_initialized
    assert c.function_name == "f_default"
    assert not len(c.class_name)
    assert "f_default" in c.source_code and "directory = " in c.source_code
    assert "f_default" in c.object_code and not "directory" in c.object_code
    assert "test" in c.defaults
    assert sum([True for i in ["x", "test"] if i in c.co_vars])
    assert {"x" : "test", "test" : None} == c.input_params

    # test no defaults
    x = f_nodefault

    c = Code(x)
    assert not c.is_class
    assert c.is_function
    assert c.is_callable
    assert not c.is_initialized
    assert c.function_name == "f_nodefault"
    assert not len(c.class_name)
    assert "f_nodefault" in c.source_code and "directory = " in c.source_code
    assert "f_nodefault" in c.object_code and not "directory" in c.object_code
    assert len(c.defaults) == 0
    assert sum([True for i in ["x", "test"] if i in c.co_vars])
    assert not len(c.input_params)


    c1 = Code(f_nodefault)
    c2 = Code(f_nodefault)
    assert c1 == c1
    assert c1 == c2

    c1 = Code(f_default)
    assert c1 != c2

    x = set([c1, c2, c1, c2])
    assert len(x) == 2

def test_merge_data():
    x1 = {"All": [2], "a": 1, "b": {"test1": 0}}
    x2 = {"All": [1], "a": 2, "b": {"test2": 0}}
    x_t = {"All": [2, 1], "a": 3, "b": {"test1": 0, "test2": 0}}
    T = Tools()
    out = T.MergeData(x1, x2)
    assert out == x_t

    x1 = {"a": 1, "b": {"test1": 0}}
    x2 = {"All": [1], "a": 2, "b": {"test2": 0}}
    x_t = {"All": [1], "a": 3, "b": {"test1": 0, "test2": 0}}
    T = Tools()
    out = T.MergeData(x1, x2)
    assert x_t == out

def test_data_merging():
    T = Tools()
    d = {
        "1": [["1", "2"], ["3", "4"], ["5", "6"], [["2"]]],
        "2": ["5", "6"],
        "3": [["1", "2"], ["3", "4"]],
    }
    assert T.MergeListsInDict(d) == [
        "1", "2", "3", "4", "5", "6", "2",
        "5", "6", "1", "2", "3", "4",
    ]

    l = [["1", "2"], ["3", "4"], ["5", "6"], [["2"]]]
    assert T.MergeNestedList(l) == ["1", "2", "3", "4", "5", "6", "2"]


def test_ls_files():
    I = Tools()
    F = I.lsFiles(directory)
    for i in range(3):
        assert directory + "Dir" + str(i + 1) in F
    assert len(I.lsFiles(directory + "Dir1", ".txt")) == 3

def test_ls():
    I = Tools()
    F = I.lsFiles(directory)
    for i in range(3):
        assert directory + "Dir" + str(i + 1) in F
    if I.ls("FakeDirectory") != []:
        return False

def test_list_files_in_dir():
    D = {
        directory + "Dir1": ["1.txt"],
        directory + "Dir2": "2.txt",
        directory + "Dir3/": "*",
    }
    I = Tools()
    O = I.ListFilesInDir(D, ".txt")
    assert "1.txt" in O[I.abs(directory + "Dir1")]
    assert "2.txt" in O[I.abs(directory + "Dir2")]
    assert "1.txt" in O[I.abs(directory + "Dir3")]
    assert "2.txt" in O[I.abs(directory + "Dir3")]
    assert "3.txt" in O[I.abs(directory + "Dir3")]


def test_is_file():
    I = Tools()
    for i in range(3):
        pth = directory + "Dir" + str(i + 1)
        pth += "/" + str(i + 1) + ".txt"
        assert I.IsFile(pth)
    assert I.IsFile(directory + "Dir1") == False

def test_pickle():
    x = {"...": ["test"]}
    PickleObject(x, "Test")
    p = UnpickleObject("Test")
    assert x == p
    T = Tools()
    T.rm("./_Pickle")


if __name__ == "__main__":
    test_code_extraction()
    test_merge_data()
    test_data_merging()
    test_ls_files()
    test_list_files_in_dir()
    test_is_file()
    test_pickle()
    # test_source_code_extraction()
