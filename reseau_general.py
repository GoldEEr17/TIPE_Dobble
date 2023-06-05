
import numpy as np
from time import time
import random
import pickle
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk


class Reseau():
    def __init__(self,taille:[int],act_function='sigmoid',input_parameters=None,stored_dataset=[]):
        self.taille = taille
        self.act_function = act_function
        self.stored_dataset = stored_dataset
        self.symbols = {} # dictionnaire associant à chaque symbole (écrit en string) son nombre de représentations dans data_set
        self.weights = [[]] + [np.random.uniform(-3,3,(self.taille[i],self.taille[i-1])) for i in range(1,len(self.taille))]
        self.biases = [[]] + [np.random.uniform(-2,2,(self.taille[i],1)) for i in range(1,len(self.taille))]
        if not input_parameters is None :
            self.weights = input_parameters[0]
            self.biases = input_parameters[1]

        self.manual_stop = False
        self.accuracy = -1
        self.average_cost = -1
        self.total_generations = 0
        self.total_duration = 0


    def activation_function(self,t):
        if self.act_function == 'sigmoid' :
            return 1 / (1+np.exp(-t))

    def der_activation_function(self,t):
        if self.act_function == 'sigmoid' :
            return np.exp(-t) / (1+np.exp(-t))**2

    def normalized(self,DATA):
        # if isinstance(DATA[-1][0], list) or len(DATA[-1][0].shape) != 2 :
        if isinstance(DATA[-1][0],np.ndarray) and len(DATA[-1][0].shape) == 2 : return DATA
        for i in range(len(DATA)) :
            DATA[i][0] = np.reshape(DATA[i][0],(-1,1)).astype(float)
            DATA[i][1] = np.reshape(DATA[i][1],(-1,1)).astype(float)

        return DATA

    def shuffled_data(self,DATA,data_shrink,data_number):
        nb_elements = min(data_number, data_shrink*len(DATA),)
        return random.sample(DATA,nb_elements)

    def string_duration(self,seconds):
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, seconds = seconds // 60, seconds % 60
        if minutes < 60:
            return f"{minutes}min {seconds}s"
        hours, minutes = minutes // 60, minutes % 60
        return f"{hours}h {minutes}min"

    def set_stop(self):
        self.manual_stop = True


    def evaluate_accuracy(self,data=None,do_print=False):
        if data is None :
            if len(self.stored_dataset) == 0 : raise Exception("pas de stored_dataset, impossible de calculer l'accuracy")
            data = self.stored_dataset
        data = self.normalized(data)
        correct_results = 0

        for entree, sortie_attendue in data:
            sortie = self.feedforward(entree)
            smoothed_sortie = [0 if x<0.5 else 1   for x in sortie]
            smoothed_sortie_attendue = [0 if x<0.5 else 1   for x in sortie_attendue]
            if smoothed_sortie == smoothed_sortie_attendue :
                correct_results += 1

        self.accuracy = correct_results / len(data)
        if do_print==True : print(self.accuracy)


    def evaluate_cost(self,data=None,do_print=False): # renvoie la moyenne des coûts C0 pour chaque exemple (i.e. somme des carrés des écarts de chaque neuronne de la dernière couche)
        if data is None :
            if self.stored_dataset == [] : raise Exception('pas de stored_dataset, impossible de calculer la précision')
            data = self.stored_dataset
        data = self.normalized(data)

        total_cost = 0.0
        for entree,sortie_attendue in data :
            sortie = self.feedforward(entree)
            for a_calc,a_att in zip(sortie[1],sortie_attendue[1]) :
                total_cost += (a_calc - a_att)**2

        average_cost = total_cost / len(data) / self.taille[-1]  # coût moyen par neuronne final par exemple
        self.average_cost = average_cost

        if do_print == True :
            print(f"le côut moyen par neuronne final par exemple est de {self.average_cost}")


    def feedforward(self,A_K:[float],J=1): # entrée A_K : vecteur colonne des activations de la couche K, précédent la couche suivante J, on va jusqu'à la dernière couche récursivement
        # if isinstance(A_K,list) or len(A_K.shape) !=2 : raise ValueError('PAS un vecteur colonne en entrée de feedforward')
        A_K = np.reshape(A_K,(-1,1))
        if J == len(self.taille) :
            return A_K
        else :
            W_J = self.weights[J]
            B_J = self.biases[J]
            Z_J = np.dot(W_J, A_K) + B_J
            A_J = self.activation_function(Z_J)

            return self.feedforward(A_J,J+1)

    def all_activations(self,entree):
        entree = np.reshape(entree,(-1,1))
        A_K, J, all_A,all_Z = entree, 1, [entree], [[]]
        while J < len(self.taille) :
            W_J = self.weights[J]
            B_J = self.biases[J]
            Z_J = np.dot(W_J,A_K) + B_J
            A_K = self.activation_function(Z_J)

            all_A.append(A_K)
            all_Z.append(Z_J)
            J += 1
        return [all_A,all_Z]


    def gradient_local(self,entree,sortie_attendue): # entree et sortie_attendue doivent être des vecteurs colonnes
        entree = np.reshape(entree,(-1,1))
        sortie_attendue = np.reshape(sortie_attendue,(-1,1))


        partial_derivatives_weights = [np.zeros_like(self.weights[K]) for K in range(len(self.taille))]
        partial_derivatives_biases = [np.zeros_like(self.biases[K]) for K in range(len(self.taille))]

        all_A, all_Z = self.all_activations(entree)
        sortie = all_A[-1]

        par_der_A_S = par_der_A_K = np.array([ 2 * 1 * (sortie[k,0] - sortie_attendue[k,0])   for k in range(len(sortie))   ]).reshape(-1,1)
        J = len(self.taille)-1
        while J > 0 :
            K = J-1
            par_der_A_J = par_der_A_K

            # Activations : d a_k0 = sum(i=1 à l(J) de drond_cout a_j * σ'(z_j) * w_jk0 )
            par_der_A_K = np.zeros_like(all_A[K])
            for j in range(len(par_der_A_J)) :
                par_der_A_K += par_der_A_J[j] * self.der_activation_function(all_Z[J][j]) * self.weights[J][j,np.newaxis].T

            # Biais : d b_j0 = drond_cout a[_j0 * σ'(z_j0) * 1
            partial_derivatives_biases[J] += par_der_A_J * self.der_activation_function(all_Z[J]) * 1

            # Poids : d w_j0k0 = drond_cout a_j0 * σ'(z_j0) * a_k0
            partial_derivatives_weights[J] += all_A[K].T # On place d'abord a_k dans chaque colonne k
            partial_derivatives_weights[J] *= par_der_A_J * self.der_activation_function(all_Z[J]) # on multiplie ensuite chaque ligne j par drond a_j * σ'(z_j)

            # Récursivement, on calcule maintenant les dérivées partielles des poids associés à la couche précédente (en s'appuyant sur les dérivées partielles des activations de cette couche)
            J -= 1


        return [partial_derivatives_weights,partial_derivatives_biases]



    def apprendre(self,DATA:[ [[float],[float]] ]=None,pas=10.0, affichage_graphique=True, data_shrink=1.0,data_number=None,precision=1.01,duration=999999999,nb_generations=999999999,CHECK_PROPORTION=0.1):
        if DATA is None : DATA = self.stored_dataset
        if data_number is None : data_number = len(DATA)
        gen_count,start_time,elapsed_time, DATA,nb_seconds = 0,time(), 0, self.normalized(DATA),0
        if duration == 999999999 and nb_generations == 999999999 : raise Exception("pas de fin d'apprentissage !")

        # Gère la partie affichage
        if affichage_graphique == True :
            window = ttk.Window()

            gen_count_tk = tk.StringVar(value='0 générations')
            av_cost_tk = tk.StringVar(value=f'coût moyen : {self.average_cost}')


            global_frame = ttk.Frame(master=window)
            global_frame.pack()

            stop_button = ttk.Button(master=global_frame,text='stop',command= self.set_stop)
            stop_button.pack()

            gens_text = ttk.Label(master=global_frame,textvariable=gen_count_tk)
            gens_text.pack()

            av_cost_text = ttk.Label(master=global_frame,textvariable=av_cost_tk)
            av_cost_text.pack()



            window.update()

        def criteria_reached():
            nonlocal gen_count, elapsed_time,nb_seconds
            # if np.random.random() <= CHECK_PROPORTION :
            #     self.evaluate_accuracy(DATA)
            gen_count += 1
            elapsed_time = time() - start_time
            if int(elapsed_time) > nb_seconds : # mises à jour 1fois par seconde
                nb_seconds = int(elapsed_time)
                print(f'{nb_seconds}s passées')
                self.evaluate_accuracy(DATA)
                self.evaluate_cost(DATA)
                if affichage_graphique :
                    # gen_count_tk.set(f'{gen_count} générations')
                    av_cost_tk.set(f'coût moyen : {self.average_cost}')
                    pass

            if affichage_graphique :
                gen_count_tk.set(f'{gen_count} générations')

                window.update()

            return gen_count >= nb_generations or elapsed_time >= duration or self.manual_stop

        while not criteria_reached() :

            data = self.shuffled_data(DATA,data_shrink,data_number)

            G_global_W = [np.zeros_like(self.weights[K]) for K in range(len(self.taille))]
            G_global_B = [np.zeros_like(self.biases[K]) for K in range(len(self.taille))]
            for entree, sortie_attendue in data :
                G_local_W, G_local_B = self.gradient_local(entree,sortie_attendue)
                G_global_W = [arr_glob + arr_loc for arr_glob,arr_loc in zip(G_global_W,G_local_W)]
                G_global_B = [arr_glob + arr_loc for arr_glob,arr_loc in zip(G_global_B,G_local_B)]

            # print(f"gen{gen_count} : gradient_weights=\n {G_global_W[1]}")
            # print(f"gen{gen_count} : gradient_biases=\n {G_global_B[1]}")

            self.weights = [arrW  +  pas * (-g_global_W) / len(data)    for arrW,g_global_W in zip(self.weights,G_global_W)]
            self.biases = [arrB + pas * (-g_global_B) / len(data)  for arrB, g_global_B in zip(self.biases,G_global_B)]



        print('finished working !')
        # self.evaluate_accuracy(DATA)
        self.manual_stop = False
        self.total_generations += gen_count
        self.total_duration += nb_seconds
        print(f'\nnb générations faites : {gen_count} ({self.total_generations}) | durée : {self.string_duration(elapsed_time)}')

        if affichage_graphique :
            window.mainloop()



