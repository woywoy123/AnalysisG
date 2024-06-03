from AnalysisG.IO import IO

root1 = "./samples/sample1/smpl1.root"
root2 = "./samples/sample1/smpl2.root"
root3 = "./samples/sample1/*"

def test_reading_root():
    x = IO()
    x.Files = [root3]
    x.scan_keys()

if __name__ == "__main__":
    test_reading_root()
