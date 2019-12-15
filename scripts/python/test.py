import numpy as np

t1 = np.arange(1, 4)
t2 = np.arange(2, 5)
t3 = [t1, t2]

t4 = np.random.randint(5, size=(3, 2))

t5 = t4[:,1]
print t4
print t5

print np.arange(0, 3*2).reshape([3,2])
