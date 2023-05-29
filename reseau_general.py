
import numpy as np
from time import time



class Reseau():
    def __init__(self,taille:[int]):
        self.taille = taille
        # self.poids = [[]] + [  [ np.ones((taille[i],taille[i-1])),np.ones((taille[i],1)) ]  for i in range(1,len(taille))  ]
        self.poids = [[]] + [ [ np.random.uniform(-3,3,(taille[i],taille[i-1])),  np.random.uniform(-2,2,(taille[i],1))  ]      for i in range(1,len(self.taille)) ]
        self.accuracy = -1
        self.total_generations = 0


    def sigmoid(self,t):
        return 1 / (1+np.exp(-t))

    def shuffled_data(self,DATA,data_shrink,data_number):
        nb_elements = min(data_number, data_shrink*len(DATA),)
        return np.random.sample(DATA,nb_elements)

    def evaluate_accuracy(self,data):
        for entree, sortie_attendue in data:
            sortie = self.feedforward(entree)
            correct_results = 0
            if max(range(len(sortie)), key=lambda i: sortie[i]) == max(range(len(sortie_attendue)), key=lambda i: sortie_attendue[i]):
                correct_results += 1
        self.accuracy = correct_results / len(data)



    def feedforward(self,A_K:[float],J:int): # entrée A_K : vecteur colonne des activations de la couche K, précédent la couche suivante J, on va jusqu'à la dernière couche récursivement
        if len(A_K.shape) == 1 : # s'assure que A_K soit bien un vecteur colonne et non une liste pour éviter les soucis
            A_K = A_K[:,np.newaxis]
        if J == len(self.taille) :
            return A_K
        else :
            W_J = self.poids[J][0]
            B_J = self.poids[J][1]
            Z_J = np.dot(W_J, A_K) + B_J #Commentaire : le produit matrice/vecteur se fait que A_K soit un vecteur colonne ou une liste isomorphe, pas de pb après
            A_J = self.sigmoid(Z_J)

            self.feedforward(A_J,J+1)


    def gradient_local(self,entree,sortie_attendue):
        partial_derivatives = np.zeros_like(self.poids)


        def backpropagation(ratio_A_J,K):
            if K == -1 :
                pass
            else :




    def apprendre(self,DATA:[ [[float],[float]] ],pas=1.0, data_shrink=1.0,data_number=None,precision=0,time=0,nb_generations=0,CHECK_PROPORTION=0.1):

        gen_count,start_time,elapsed_time,manual_stop = -1,time(), 0, False
        def criteria_reached():
            if np.random.random() <= CHECK_PROPORTION :
                self.evaluate_accuracy(DATA)
            gen_count += 1
            elapsed_time = time() - start_time
            # manual stop^with interface ?

            return self.accuracy >= precision or gen_count >= nb_generations or elapsed_time >= time or manual_stop


        while not (criteria_reached()) :

            data = self.shuffled_data(DATA,data_shrink,data_number)

            gradient_global = np.zeros_like(self.poids)
            for entree, sortie_attendue in data :
                gradient_global += self.gradient_local(entree,sortie_attendue)

            self.poids += pas * gradient_global

        self.evaluate_accuracy(DATA)
        self.total_generations += nb_generations






my_reseau = Reseau([6,4,4,3])
# print(my_reseau.poids)

my_entree = np.array([0.7,0.4,0.6,0.8,0.1,0.1])

# print(my_entree.shape)
# my_entree = my_entree[:,np.newaxis]
# print(my_entree.shape)

print(my_reseau.feedforward(my_entree,1))

# my_reseau.apprendre([True,False,False],100.5)

