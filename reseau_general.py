
import numpy as np

class Reseau():
    def __init__(self,taille):
        self.taille = taille
        self.poids = [[]] + [  [ np.ones((taille[i],taille[i-1])),np.ones((taille[i],1)) ]  for i in range(1,len(taille))  ]

    def sigmoid(self,t):
        return 1 / (1+np.exp(-t))


    def feedforward(self,A_K,J): # entrée A_K : vecteur colonne des activations de la couche K, précédent la couche suivante J, on va jusqu'à la dernière récursivement
        if J == len(self.taille) :
            print("finished", A_K)
            return A_K
        else :
            W_J = self.poids[K][0]
            B_J = self.poids[K][1]
            Z_J = np.dot(W_J, A_K) + B_J
            A_J = self.sigmoid(Z_J)

            self.feedforward(A_J,J+1)





my_reseau = Reseau([6,4,4,3])
# print(my_reseau.poids)

my_entree = np.array([0.7,0.4,0.6,0.8,0.1,0.1])

print(my_entree.shape)
my_entree = my_entree[:,np.newaxis]
print(my_entree.shape)

print(my_reseau.feedforward(my_entree,1))


