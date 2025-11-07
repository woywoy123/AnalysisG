metrics.h
=========

**File Path**: ``modules/metrics/include/metrics/metrics.h``

**File Type**: H (Header)

**Lines**: 138

Dependencies
------------

**Includes**:

- ``TCanvas.h``
- ``TGraph.h``
- ``TH1F.h``
- ``TLegend.h``
- ``TMultiGraph.h``
- ``TStyle.h``
- ``notification/notification.h``
- ``pyc/pyc.h``
- ``structs/report.h``
- ``structs/settings.h``
- ``templates/model_template.h``

Classes
-------

``metrics``
~~~~~~~~~~~

**Inherits from**: ``tools, 
    public notification``

**Methods**:

- ``void dump_plots(int k)``
- ``void dump_loss_plots(int k)``
- ``void dump_accuracy_plots(int k)``
- ``void dump_mass_plots(int k)``
- ``void capture(mode_enum, int kfold, int epoch, int smpl_len)``
- ``void build_th1f_loss(std::map<std::string, std::tuple<torch::Tensor*, l...)``
- ``void add_th1f_loss(std::map<std::string, torch::Tensor>* type, 
     ...)``
- ``void build_th1f_accuracy(std::map<std::string, std::tuple<torch::Tensor*, l...)``
- ``void add_th1f_accuracy(torch::Tensor* pred, torch::Tensor* truth, 
      ...)``
- ``void build_th1f_mass(std::string var_name, graph_enum typ, int kfold)``

Structs
-------

``analytics_t``
~~~~~~~~~~~~~~~

**Members**:

- ``model_template* model = nullptr``
- ``model_report* report = nullptr``
- ``int this_epoch = 0``
- ``std::map<mode_enum, std::map<std::string, TH1F*>> loss_graph = {``

