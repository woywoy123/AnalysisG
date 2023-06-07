### Model Submission Directory:
- This directory is used to run models that need to be evaluated. 


### Training Parameters:
# Name | Optimizer | Parameters | Scheduler | Parameters | BatchSize | Parameter
- MRK1: ADAM | lr: 0.001, wd: 0.001 | None | None | Yes | 10
- MRK2: ADAM | lr: 0.001, wd: 0.001 | None | None | Yes | 50
- MRK3: ADAM | lr: 0.001, wd: 0.001 | None | None | Yes | 100
- MRK4: ADAM | lr: 0.001, wd: 0.001 | None | None | Yes | 200

- MRK5: ADAM | lr: 0.0001, wd: 0.001 | None | None | Yes | 10
- MRK6: ADAM | lr: 0.001, wd: 0.0001 | None | None | Yes | 50
- MRK7: ADAM | lr: 0.0001, wd: 0.0001 | None | None | Yes | 100
- MRK8: ADAM | lr: 0.0001, wd: 0.0001 | None | None | Yes | 200

- MRK9: ADAM | lr: 0.0001, wd: 0.001 | ExponentialLR | 0.5 | Yes | 10
- MRK10: ADAM | lr: 0.001, wd: 0.0001 | ExponentialLR | 1.0 | Yes | 50
- MRK11: ADAM | lr: 0.0001, wd: 0.0001 | ExponentialLR | 2.0 | Yes | 100
- MRK12: ADAM | lr: 0.0001, wd: 0.0001 | ExponentialLR | 4.0 | Yes | 200

- MRK13: ADAM | lr: 0.0001, wd: 0.001  | CyclicLR | base_lr : 0.00001, max_lr : 0.0001 | Yes | 10
- MRK14: ADAM | lr: 0.001, wd: 0.0001  | CyclicLR | base_lr : 0.00001, max_lr : 0.001 | Yes | 50
- MRK15: ADAM | lr: 0.0001, wd: 0.0001 | CyclicLR | base_lr : 0.00001, max_lr : 0.01 | Yes | 100
- MRK16: ADAM | lr: 0.0001, wd: 0.0001 | CyclicLR | base_lr : 0.00001, max_lr : 0.1 | Yes | 200

- MRK17: SGD | lr: 0.0001, wd: 0.001, momentum: 0.0001 | None | None| Yes | 10
- MRK18: SGD | lr: 0.001, wd: 0.0001, momentum: 0.0005 | None | None | Yes | 50 
- MRK19: SGD | lr: 0.0001, wd: 0.0001, momentum: 0.001 | None | None | Yes | 100 
- MRK20: SGD | lr: 0.0001, wd: 0.0001, momentum: 0.0015 | None | None | Yes | 200 

