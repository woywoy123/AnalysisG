.. _settings-struct:

settings_t Struct Attributes
----------------------------

The `settings_t` struct is mainly accessible via the `Analysis` interface and is only relevant for interfacing with the framework in `C++`.

.. cpp:struct:: settings_t

    .. cpp:var:: std::string output_path

    .. cpp:var:: std::string run_name
    
    .. cpp:var:: int epochs

    .. cpp:var:: int kfolds

    .. cpp:var:: std::vector<int> kfold

    .. cpp:var:: int num_examples

    .. cpp:var:: float train_size
   
    .. cpp:var:: bool training

    .. cpp:var:: bool validation

    .. cpp:var:: bool evaluation

    .. cpp:var:: bool continue_training

    .. cpp:var:: std::string training_dataset

    .. cpp:var:: std::string var_pt

    .. cpp:var:: std::string var_eta

    .. cpp:var:: std::string var_phi

    .. cpp:var:: std::string var_energy

    .. cpp:var:: std::vector<std::string> targets
   
    .. cpp:var:: int nbins

    .. cpp:var:: int refresh

    .. cpp:var:: int max_range

    .. cpp:var:: bool debug_mode

    .. cpp:var:: int threads
