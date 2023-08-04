Neutrino Reconstruction
***********************
The aim of this study is to benchmark the performance of the neutrino reconstruction algorithm found under [1].

Selection Criteria:
___________________

Events are required to have no more than two leptonically decaying tops. 
These are filtered at truth level (TruthChildren). 
The neutrino algorithm ``zero`` value is being set to 1e-10, since anything below this value creates large differences between C++ and Python floating point definitions.

Particle Definitions:
_____________________

Leptons and neutrinos are defined as:
- electrons
- muons 
- taus

[1]. **Analytic solutions for neutrino momenta in decay of top quarks** (arXiv: 1305.1878v2)
