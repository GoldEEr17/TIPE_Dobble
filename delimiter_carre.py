from pylab import *
import os
import random



N1 = 100

M_test = np.zeros((N1,N1))

for i in range(N1):
    for j in range(N1):

        if random.
        M_test[i,j] = random.randint(0,1)


print(M_test)


figure()
# axis('off')

imshow(M_test,interpolation='nearest',cmap='gray')

show()