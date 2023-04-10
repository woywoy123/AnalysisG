import Event
from time import time

x = Event.CyEvent()
x.EventIndex = 1
x.Hash = "hello world!"
t1 = time()
n = 100000
for i in range(n):
    x.Hash
t2 = time();
print((t2 - t1)/n)
