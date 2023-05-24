from pylab import *
import os
import random

print(os.listdir())



### ZONE DE TESTS





# un_dict = {'plouf':[2,3]}
# un_dict['plouf'][1] -= 5
# print(un_dict)

# print(np.zeros((2,7)).shape  )

# a_dict = {8: [5,7], 2:[8,9]}
# for x,y in a_dict.values() :
#     print(x)
#     print(y)



### CODE








def matrice_random(): #carré 100x100
    N1 = 100
    M_test = np.zeros((N1,N1))

    for i in range(N1):
        for j in range(N1):

            if random.random() >=0.9 :
                M_test[i,j] = 1
            else :
                M_test[i,j] = 0

    return M_test




def affiche(MATRICE):
    figure()
    # axis('off')
    imshow(MATRICE,interpolation='nearest',cmap='BrBG') # ou binary pour blanc=0 noir = 1
    show()



affiche(matrice_random())

















## Fonction : part d'un point (i0,j0) et étend un carré (après: rectangle) jusqu'à ce qu'il n'y ait plus aucun
## 1 en bordure de sa zone










def delimite_apd_point(M,point_case) :


    p, q = M.shape
    i0,j0 = point_case

    # Contour est une liste de 4 couples donnant les 4 coins du rectangle/carré
    # On pourrait aussi faire un carré de centre bidule ou rectangle de longueur/largeur et de centre machun, bof pour étendre


    Contour = {'hg':[i0,j0], 'hd':[i0,j0],'bg':[i0,j0],'bd':[i0,j0]}


    def test_depasse() :
        for x,y in Contour.values() :
            print(x,y)
            if x < 0 or x > p or y < 0 or y > q :
                return True
        return False


    # def depasse_gauche


    def b_gauche():
        CONST = Contour['hg'][1]
        debut = Contour['hg'][0]
        fin = Contour['bg'][0]

        print("pré passage gauche")
        for k in range(debut,fin+1) :
            print("1 passage gauche")
            if M[k,CONST] == 1 :
                return True
        return False


    def b_droite():
        CONST = Contour['hd'][1]
        debut = Contour['hd'][0]
        fin = Contour['bd'][0]

        for k in range(debut,fin+1) :
            if M[k,CONST] == 1 :
                return True
        return False

    def b_haut():
        CONST = Contour['hg'][0]
        debut = Contour['hg'][1]
        fin = Contour['hd'][1]

        for k in range(debut,fin+1) :
            if M[k,CONST] == 1 :
                return True
        return False

    def b_haut():
        CONST = Contour['bg'][0]
        debut = Contour['bg'][1]
        fin = Contour['bd'][1]

        for k in range(debut,fin+1) :
            if M[k,CONST] == 1 :
                return True
        return False






    while (not test_depasse) and (b_gauche() or b_droite() or b_haut() or b_bas()) :

        if b_gauche() :
            Contour['hg'][1] -= 1
            Contour['bg'][1] -= 1

        if b_droite() :
            Contour['hd'][1] += 1
            Contour['bd'][1] += 1

        if b_haut() :
            Contour['hg'][0] -= 1
            Contour['hd'][0] -= 1

        if b_bas() :
            Contour['bg'][1] += 1
            Contour['bd'][1] += 1



    def afficher_resultat() :

        E = M.copy() # E pour encadrée : but = afficher image avec le cadre choisit
        haut = Contour['hg'][0]
        bas = Contour['bg'][0]
        gauche = Contour['hg'][1]
        droite = Contour['hd'][1]


        for k in range(gauche,droite+1):
            E[k,haut] = 1/2
        for k in range(gauche,droite+1):
            E[k,bas] = 1/2
        for k in range(haut,bas+1):
            E[k,gauche] = 1/2
        for k in range(haut,bas+1):
            E[k,droite] = 1/2

        affiche(E)


    afficher_resultat()








from delimiter_symboles import M_import1



M4 = matrice_random()
M4[50,50] = 1
affiche(M4)

delimite_apd_point(M4,(50,50))




