# FourTops
This repo will be used to test the read and write of the FourTops data samples that were created using the Beyond Standard Model group.

Below is a list of tags for each signal sample each of the samples have different mass points as shown below:  

```
DSID	    Mass        Width       Cross-section for ct=1 [pb]
310845	  1 TeV       auto               3.99e-03	 
313346	  1.25 TeV    auto               1.2703e-03	 
310846	  1.5 TeV     auto               4.49e-04	 
310847	  2 TeV       auto               7.02e-05	 
313180	  2.5 TeV     auto               1.3395e-05	 
313181	  3 TeV       auto               2.9508e-06
```

## Some Simple Documentation for the Code: 
- Simply clone the repo 
- ```python main.py ```

What the code is capable of doing so far:
- Match the Signal tops to generate a clear resonance.
- Match the Children of Signal Tops and generate resonance very much identical to the absolute truth top branches 
- Match the truth jets to the children (using delta R) and then plotting the resonance 
- The code is highly optimized to run with multicore. This means reading branches is much faster and converting branches from AWKWARD array to numpy is significantly faster
- Under main.py there are several closure tests that illustrate the above 

New Core Refactored Code:
- Particles/Jet Particles are represented as objects (allows for much easier multithreading and later book keeping)
- Alerting and debugging classes
- Event Compiler and variable manager makes it much easier to deal with branches and significantly decreases processing time

Still to do:
- Improve the UpRoot reader and include Debugging and Alerting class
- Remove legacy code (SignalSpectator)
- Match observed detector particles to truth jets and children of signal tops 


