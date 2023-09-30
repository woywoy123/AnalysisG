.. _meta-data:

MetaData Wrapper
****************

.. py:class:: MetaData

    This is a class which wraps the output of ``PyAmi`` when performing a meta-data search on the input samples.
    To use this wrapper effectively, you need to authenticate using the ``voms-proxy-init`` command.
    Luckily, the framework comes with a pre-built script, which does this for you, simply enter ``auth_pyami`` into the terminal and it will ask for the ``userkey/usercert.pem`` directories, followed by a password prompt. 
    If the user is not authenticated, this step is skipped and the framework will attempt to inspect the sample's available meta-data. 
   
    :param str file: The input path of the file.
    :param bool scan_ami: A switch which enables/disables the meta scanning.
    :param str sampletype: An option which narrows down the DAOD sample search to a specific group.

    .. py:method:: IndexToSample(int index) -> str

    .. py:method:: ProcessKeys(dict val) -> None 

    .. py:attribute:: Trees -> list

        Expects a list of strings pointing to the trees to be read.

    .. py:attribute:: Branches -> list

        Expects a list of strings pointing to the branches to be read.

    .. py:attribute:: Leaves -> list
        
        Expects a list of strings pointing to any leaves to be read.

    .. py:attribute:: original_input -> str
        
        Returns the original file name, i.e. file.

    .. py:attribute:: dsid -> int

        Returns the dsid of the data set matched to this ROOT file.

    .. py:attribute:: amitag -> str

        Returns the amitag of the same being used, this can be retrieved within the sample's metadata.

    .. py:attribute:: generators -> str

        Returns the generator which produced this sample.

    .. py:attribute:: isMC -> bool

        Returns a boolean to indicate whether the sample is Monte Carlo

    .. py:attribute:: derivationFormat -> str

    .. py:attribute:: eventNumber -> int

    .. py:attribute:: ecmEnergy -> float

    .. py:attribute:: genFiltEff -> float

    .. py:attribute:: completion -> float

    .. py:attribute:: beam_energy -> float

    .. py:attribute:: crossSection -> float

    .. py:attribute:: crossSection_mean -> float

    .. py:attribute:: totalSize -> float

        Returns the total file size (memory)

    .. py:attribute:: nFiles -> int

        Returns the number of files within this sample set

    .. py:attribute:: run_number -> int

    .. py:attribute:: totalEvents -> int

    .. py:attribute:: datasetNumber -> int

    .. py:attribute:: identifier -> str

    .. py:attribute:: prodsysStatus -> str

    .. py:attribute:: dataType -> str

    .. py:attribute:: version -> str

    .. py:attribute:: PDF -> str

    .. py:attribute:: AtlasRelease -> str

    .. py:attribute:: principalPhysicsGroup -> str

    .. py:attribute:: physicsShort -> str

    .. py:attribute:: generatorName -> str

    .. py:attribute:: geometryVersion -> str

    .. py:attribute:: conditionsTag -> str

    .. py:attribute:: generatorTune -> str

    .. py:attribute:: amiStatus -> str

    .. py:attribute:: beamType -> str

    .. py:attribute:: productionStep -> str

    .. py:attribute:: projectName -> str

    .. py:attribute:: statsAlgorithm -> str

    .. py:attribute:: genFilterNames -> str

    .. py:attribute:: file_type -> str

    .. py:attribute:: DatasetName -> str

    .. py:attribute:: event_index -> int

    .. py:attribute:: original_name -> str

    .. py:attribute:: original_path -> str

    .. py:attribute:: hash -> str

    .. py:attribute:: keywords -> list

    .. py:attribute:: weights -> list

    .. py:attribute:: keyword -> list

    .. py:attribute:: found -> bool

    .. py:attribute:: config -> dict

    .. py:attribute:: GetLengthTrees -> dict

    .. py:attribute:: MissingTrees -> list

    .. py:attribute:: MissingBranches -> list

    .. py:attribute:: MissingLeaves -> list

    .. py:attribute:: DAODList -> list

    .. py:attribute:: Files -> dict

    .. py:attribute:: DAOD -> str

    .. py:attribute:: fileGUID -> dict

    .. py:attribute:: events -> dict

    .. py:attribute:: fileSize -> dict

    .. py:attribute:: sample_name -> str

    .. py:attribute:: logicalDatasetName -> str


UpROOT wrapper
**************

.. py:class:: UpROOT

    This class is predominantly designed to be interfaced with core modules in the framework. 
    However, it can be used as a completely standalone module with minimal configuration. 
    Part of this module is the so called `MetaData` object. 
    This class contains additional information about the input ROOT samples if ``PyAMI`` is enabled. 
    If ``PyAMI`` is not installed or authenticated to, then it will try to scrape the ROOT files for additional meta data. 


    :param Union[list, dict, str, None] ROOTFiles: Input samples
    :param Union[None, EventGenerator] EventGenerator: A switch which enables/disables the meta scanning.

    .. py:attribute:: Verbose -> int

        Changes the verbosity of the key scannig and sample detection.

    .. py:attribute:: StepSize -> int 

        Changes the cache step size within **uproot**. 

    .. py:attribute:: Threads -> int

        Sets the number of threads to utilize during the scanning process.

    .. py:attribute:: Trees -> list

        Trees to retrieve from the ROOT sample.

    .. py:attribute:: Branches -> list

        Branches to retrieve and match for the given trees.

    .. py:attribute:: Leaves -> list

        Leaves to retrieve and match for the given trees and branches.

    .. py:attribute:: Files -> dict

    .. py:attribute:: EnablePyAMI -> bool 

        Enable or disable MetaData.

    .. py:method:: GetAmiMeta() -> MetaData

    .. py:method:: ScanKeys()

        A function which scans the keys within the sample recursively and matches them with the input values.

    .. py:method:: InputSamples(input)

        This function will scan the given input for ROOT files. 
        If the input is a string containing the `.root` extension, then only that file will be used, otherwise it will assume the input is a directory and scan it for possible ROOT files.
        For lists, the function will assume these to be `.root` files and never directories. 
        If the input is a dictionary, then the keys can be interpreted as being directories, with values being either lists of ROOT files to read, or single ROOT file strings.

        :param Union[str, Dict, List] input: The input samples to use.
        
