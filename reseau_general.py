
import numpy as np
from time import time
import random as random


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

    def normalized(self,DATA, for_data_set=True):
        if for_data_set :
            if not isinstance(DATA[0][0], list) : return DATA

            for i in range(len(DATA)) :
                DATA[i][0] = np.reshape(DATA[i][0],(-1,1))
                DATA[i][1] = np.reshape(DATA[i][1],(-1,1))

            return DATA

    def shuffled_data(self,DATA,data_shrink,data_number):
        nb_elements = min(data_number, data_shrink*len(DATA),)
        return random.sample(DATA,nb_elements)

    #à corriger
    def evaluate_accuracy(self,data,do_print=False):
        data = self.normalized(data)
        for entree, sortie_attendue in data:
            sortie = self.feedforward(entree)
            correct_results = 0
            if max(range(sortie.size), key=lambda i: sortie[i]) == max(range(sortie_attendue.size), key=lambda i: sortie_attendue[i]):
                correct_results += 1
        self.accuracy = correct_results / len(data)
        if do_print==True : print(self.accuracy)

    def zeros_like_recursive(self,First):
        if isinstance(First,list):
            return [self.zeros_like_recursive(elem) for elem in First]
        if isinstance(First,np.ndarray):
            return np.zeros_like(First) # un array numpy ne peut contenir de listes ou de array
        else :
            return First # C'est par exemple juste un float


    def feedforward(self,A_K:[float],J=1): # entrée A_K : vecteur colonne des activations de la couche K, précédent la couche suivante J, on va jusqu'à la dernière couche récursivement
        # A_K = np.reshape(A_K,(len(A_K),1)) # transforme assurémenr en vecteur colonne ... en fait envoyer en entrée toujours des vecteurs colonnes
        if isinstance(A_K,list) or len(A_K.shape) !=2 : return "erreur dans feedforward : entree A_K n'est pas un vecteur colonne"
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


    def gradient_local(self,entree,sortie_attendue): # entree et sortie_attendue doivent être des vecteurs colonnes
        partial_derivatives = self.zeros_like_recursive(self.poids)

        all_A, all_Z = self.all_activations(entree)
        sortie = all_A[-1]

        par_der_A_S = np.array([ 2 * (sortie[k,0] - sortie_attendue[k,0])**2   for k in range(len(sortie))   ])[:,np.newaxis]
        def backpropagation(par_der_A_J,J):
            K = J-1
            if J == 0 :
                pass
            else :

                # Activations : d a_k0 = sum(i=1 à l(J) de drond_cout a_j * σ'(z_j) * w_jk0 )
                par_der_A_K = np.zeros_like(all_A[K])
                for j in range(len(par_der_A_J)) :
                    par_der_A_K += par_der_A_J[j] * self.der_activation_function(all_Z[J][j]) * self.poids[J][0][j,np.newaxis].T

                # Biais : d b_j0 = drond_cout a_j0 * σ'(z_j0) * 1
                wup = partial_derivatives[J][1]
                partial_derivatives[J][1] += par_der_A_J * self.der_activation_function(all_Z[J]) * 1

                # Poids : d w_j0k0 = drond_cout a_j0 * σ'(z_j0) * a_k0
                partial_derivatives[J][0] += all_A[K].T # On place d'abord a_k dans chaque colonne k
                partial_derivatives[J][0] = par_der_A_J * self.der_activation_function(all_Z[J]) * partial_derivatives[J][0] # on multiplie ensuite chaque ligne j par drond a_j * σ'(z_j)

                # Récursivement, on calcule maintenant les dérivées partielles des poids associés à la couche précédente (en s'appuyant sur les dérivées partielles des activations de cette couche)
                backpropagation(par_der_A_K,J-1)

        backpropagation(par_der_A_S,len(self.taille)-1)
        return partial_derivatives



    def apprendre(self,DATA:[ [[float],[float]] ],pas=1.0, data_shrink=1.0,data_number=None,precision=0,duration=99999,nb_generations=0,CHECK_PROPORTION=0.1):
        if data_number is None : data_number = len(DATA)
        gen_count,start_time,elapsed_time,manual_stop, DATA = -1,time(), 0, False, self.normalized(DATA)
        # self.normalized permet de s'assurer que tout soit de la forme de vecteur colonne
        if precision == 0 : CHECK_PROPORTION = -1

        #rajouter la précision...
        def criteria_reached():
            nonlocal gen_count
            if np.random.random() <= CHECK_PROPORTION :
                self.evaluate_accuracy(DATA)
            gen_count += 1
            elapsed_time = time() - start_time
            # manual stop^with interface ?

            return gen_count >= nb_generations or elapsed_time >= duration or manual_stop

        while not (criteria_reached()) :

            data = self.shuffled_data(DATA,data_shrink,data_number)

            gradient_global = self.zeros_like_recursive(self.poids)
            for entree, sortie_attendue in data :
                gradient_global += self.gradient_local(entree,sortie_attendue)  # soucis : s'il cpmprend les entrees de 0 et 1 comme des int, il peine à claculer
            gradient_global = gradient_global # 'besoin' de prendre la moyenne mais pb multiplication

            self.poids += pas * gradient_global

        self.evaluate_accuracy(DATA)
        self.total_generations += nb_generations






my_reseau = Reseau([6,4,4,3])
# print(my_reseau.poids)

entree1 = np.array([0.7,0.4,0.6,0.8,0.1,0.1])[:,np.newaxis]
sortie_attendue1 = np.array([1,0,0])[:,np.newaxis]

# print(entree1.shape)
# entree1 = entree1[:,np.newaxis]
# print(entree1.shape)

# print(my_reseau.feedforward(entree1,1))
# print(my_reseau.all_activations(entree1))

# my_reseau.apprendre([True,False,False],100.5)

# test_gradient_local1 = my_reseau.gradient_local(entree1,sortie_attendue1)
# Résultat : le test a l'air concluant, ça marche maintenant !

my_DATA = [ [[0,0,1.0,1,0,0],[0.0,1,0]], [[1,1,0,0.,0,0],[1.,0,0]], [[0.,0,0,0,1,1],[0.,0,1]]  ]

# print(my_reseau.evaluate_accuracy(my_DATA,do_print=True))

my_reseau.apprendre(my_DATA,nb_generations=100)



