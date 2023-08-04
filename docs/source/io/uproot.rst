UpROOT wrapper
**************
This class is predominantly designed to be interfaced with core modules in the framework. 
However, it can be used as a completely standalone module with minimal configuration. 
Part of this module is the so called `MetaData` object. 
This class contains additional information about the input ROOT samples if PyAMI is enabled. 
If PyAMI is not installed or authenticated to, then it will try to scrape the ROOT files for additional meta data. 

Functions:
__________

- ``InputSamples(input: Union[str, Dict, List])``:
    This function will scan the given input for ROOT files. 
    If the input is a string containing the `.root` extension, then only that file will be used, otherwise it will assume the input is a directory and scan it for possible ROOT files.
    For lists, the function will assume these to be `.root` files and never directories. 
    If the input is a dictionary, then the keys can be interpreted as being directories, with values being either lists of ROOT files to read, or single ROOT file strings.

Attributes:
___________

- ``Trees``: 
    Expects a list of strings pointing to the trees to be read.

- ``Branches``: 
    Expects a list of strings pointing to any branches.

- ``Leaves``: 
    Expects a list of strings pointing to any leaves to be read.

- ``ScanKeys``: 
    Will check whether the given `Trees/Branches/Leaves` are found within the ROOT samples.


