
import numpy as np
from time import time



class Reseau():
    def __init__(self,taille:[int],act_function='sigmoid'):
        self.taille = taille
        self.act_function = act_function
        self.poids = [[]] + [ [ np.random.uniform(-3,3,(taille[i],taille[i-1])),  np.random.uniform(-2,2,(taille[i],1))  ]      for i in range(1,len(self.taille)) ]
        self.accuracy = -1
        self.total_generations = 0


    def activation_function(self,t):
        if self.act_function == 'sigmoid' :
            return 1 / (1+np.exp(-t))

    def der_activation_function(self,t):
        if self.act_function == 'sigmoid' :
            return np.exp(-t) / (1+np.exp(-t))**2

    def shuffled_data(self,DATA,data_shrink,data_number):
        nb_elements = min(data_number, data_shrink*len(DATA))
        return np.random.sample(DATA,nb_elements)

    def evaluate_accuracy(self,data):
        for entree, sortie_attendue in data:
            sortie = self.feedforward(entree)
            correct_results = 0
            if max(range(len(sortie)), key=lambda i: sortie[i]) == max(range(len(sortie_attendue)), key=lambda i: sortie_attendue[i]):
                correct_results += 1
        self.accuracy = correct_results / len(data)



    def feedforward(self,A_K:[float],J:int): # entrée A_K : vecteur colonne des activations de la couche K, précédent la couche suivante J, on va jusqu'à la dernière couche récursivement
        A_K = np.reshape(A_K,(len(A_K),1)) # transforme assurémenr en vecteur colonne
        if J == len(self.taille) :
            return A_K
        else :
            W_J = self.poids[J][0]
            B_J = self.poids[J][1]
            Z_J = np.dot(W_J, A_K) + B_J #Commentaire : le produit matrice/vecteur se fait que A_K soit un vecteur colonne ou une liste isomorphe, pas de pb après
            A_J = self.activation_function(Z_J)

            self.feedforward(A_J,J+1)

    def all_activations(self,entree):
        entree = np.reshape(entree,(len(entree),1))
        A_K, J, all_A,all_Z = entree, 1, [entree], [[]]
        while J < len(self.taille) :
            W_J = self.poids[J][0]
            B_J = self.poids[J][1]
            Z_J = np.dot(W_J,A_K) + B_J  # Commentaire : le produit matrice/vecteur se fait que A_K soit un vecteur colonne ou une liste isomorphe, pas de pb après
            A_K = self.activation_function(Z_J)

            all_A.append(A_K)
            all_Z.append(Z_J)
            J += 1
        return [all_A,all_Z]




    def gradient_local(self,entree,sortie_attendue):
        partial_derivatives = np.zeros_like(self.poids)
        all_A, all_Z = self.all_activations(entree)
        #partial derivatives_A
        sortie = all_A[-1]

        par_der_A_S = [ 2 * (sortie[k] - sortie_attendue[k])**2   for k in range(len(sortie))   ]
        def backpropagation(par_der_A_J,J):
            if J == 0 :
                pass
            else :
                par_der_A_K = np.zeros_like(all_A[J-1])
                for j in range(self.taille[J]) :
                    par_der_A_K += par_der_A_J[j] * self.der_activation_function(all_Z[J][j] * self.poids[J][0][j,:].T)

                # Biais : b_j0 = drond A_j0 * sigma' z_j0 * 1
                partial_derivatives[J][1] += par_der_A_J * self.der_activation_function(all_Z[J]) * 1
                # Poids.T
                for j_row
                    partial_derivatives[J][0] +=  [j] * self.der_activation_function(all_Z[j]) * all_A[J-1].T    for j in range(self.taille[J]) for k in range(self.taille[J-1])




                backpropagation(par_der_A_K,J-1)






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
print(my_reseau.all_activations(my_entree))

# my_reseau.apprendre([True,False,False],100.5)

