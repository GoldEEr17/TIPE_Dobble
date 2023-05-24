from pylab import *
import os
import random

# print(os.listdir())



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
    imshow(MATRICE) # ou binary pour blanc=0 noir = 1 # ou BrBG
    show()



# affiche(matrice_random())

















## Fonction : part d'un point (i0,j0) et étend un carré (après: rectangle) jusqu'à ce qu'il n'y ait plus aucun
## 1 en bordure de sa zone









def delimite_apd_point(M,point_case) :


    # On convertit bien la matrice à un format (p,q) dont la valeur contenu est soit 1 (symbole) soit 0(rien).
    # Actuellement, le format RGBA fournit contient le 1 en position 1 (image en vert). En fait : plutôt gérer ceci avant d'entrer dans cette fonction, comme
    # ça on fournit juste du format (p,q) avec des 1 et des 0 et tout est clair
    if len(M.shape) > 2 and M.shape[2] == 4 :
        M = M[:,:,1]


    p, q = M.shape
    i0,j0 = point_case
    # print(i0,j0)
    # M[i0,j0] = 1/2
    # figure()
    # imshow(M)
    # show()

    # Contour est une liste de 4 couples donnant les 4 coins du rectangle/carré
    # On pourrait aussi faire un carré de centre bidule ou rectangle de longueur/largeur et de centre machun, bof pour étendre


    Contour = {'hg':[i0,j0], 'hd':[i0,j0],'bg':[i0,j0],'bd':[i0,j0]}


    def test_depasse() :
        for x,y in Contour.values() :
            if x < 0 or x > p or y < 0 or y > q :
                return True
        return False


    # def depasse_gauche


    def b_gauche():
        CONST = Contour['hg'][0]
        debut = Contour['hg'][1]
        fin = Contour['bg'][1]

        for k in range(debut,fin+1) :
            if M[CONST,k] == 1 :
                return True
        return False


    def b_droite():
        CONST = Contour['hd'][0]
        debut = Contour['hd'][1]
        fin = Contour['bd'][1]

        for k in range(debut,fin+1) :
            if M[CONST,k] == 1 :
                return True
        return False

    def b_haut():
        CONST = Contour['hg'][1]
        debut = Contour['hg'][0]
        fin = Contour['hd'][0]

        for k in range(debut,fin+1) :
            if M[k,CONST] == 1 :
                return True
        return False

    def b_bas():
        CONST = Contour['bg'][1]
        debut = Contour['bg'][0]
        fin = Contour['bd'][0]

        for k in range(debut,fin+1) :
            if M[k,CONST] == 1 :
                return True
        return False





    while (not test_depasse()) and (b_gauche() or b_droite() or b_haut() or b_bas()) :

        # print("entrée while")
        # Il faut rajouter une prise en compte des bordures de l'image...
        if b_gauche() :
            Contour['hg'][0] -= 1
            Contour['bg'][0] -= 1

        if b_droite() :
            Contour['hd'][0] += 1
            Contour['bd'][0] += 1

        if b_haut() :
            Contour['hg'][1] -= 1
            Contour['hd'][1] -= 1

        if b_bas() :
            Contour['bg'][1] += 1
            Contour['bd'][1] += 1





    def afficher_resultat() : # affiche la zone sélectionnée encadrée en rouge

        E = M.copy() # E pour encadrée : but = afficher image avec le cadre choisit
        haut = Contour['hg'][1]
        bas = Contour['bg'][1]
        gauche = Contour['hg'][0]
        droite = Contour['hd'][0]


        for k in range(gauche,droite+1):
            E[k,haut] = 1/2
        for k in range(gauche,droite+1):
            E[k,bas] = 1/2
        for k in range(haut,bas+1):
            E[gauche,k] = 1/2
        for k in range(haut,bas+1):
            E[droite,k] = 1/2

        affiche(E)

    afficher_resultat()








# from delimiter_symboles.py import D4
D4 = load('D4.npy')
D4 = D4[:,:,1]
# affiche(D4)

# print(D4.shape)
# delimite_apd_point(D4, (190,190))
# delimite_apd_point(D4, (190,125))
# delimite_apd_point(D4, (50,150))
# delimite_apd_point(D4, (150,50))
# delimite_apd_point(D4, (190,190))

# il semblerait que x et y soit inversé ce qui est étrange



D1 = load('D1.npy')
D1 = D1[:,:,1]

affiche(D1)

delimite_apd_point(D1,(210,210))






