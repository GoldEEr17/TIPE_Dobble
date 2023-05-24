from pylab import *
import os as os



'''
ZONE COMMENTAIRES DE LÉO

- tu peux utiliser "p,q = M.shape" ou "p,q = M.shape[:2]" directement ça marche. ça permet d'éviter les "lenM = M.shape" combiné avec lenM[0] pour plutôt utiliser "p, q = M.shape" puis utiliser p et q directement donc plus rapidement et plus clairement.

- J'ai proposé une version modifié de "distance" qui est littéralement identique, c'est juste pour proposer des noms de variables que je trouvep plus explicites (en guise d'exemple de à quoi c'est un peu mieux que ça ressemble)

- fonction zone : j'ai proposé une version avec noms de variables selon moi plus adaptés et plus explicites. Aussi, j'ai fait une modif d'optimisation : tester les voisins seulement si la case passe le 'test_couleur' de base. La fonction faite comme ça reste peu opti car on doit calculer chaque test_couleur de chaque case potentiellement 4 fois... Plutôt, on peut faire un passage sans test de voisins, puis un autre ou on ne fait que 'lire' (sans recalculer) la couleur des voisins pour ensuite supprimer ou non la couleur.

J'ai aussi compris qu'en fait tu ne testais même pas la couleur de la case mais seulement ses voisins, d'où maybe certains résultats bizarres enfin ça peut poser soucis éventuellement...

- proposition : adapter le programme pour compter les voisins diagonaux


'''




os.chdir('F:\TIPE Dobble clé')
# os.chdir ('U:\mateo.brun')

M1 = imread('dobble1.png')
# M2 = imread('dobble2.png')
# M3 = imread('dobble3.png')
# M4 = imread('dobble4.png')
# M5 = imread('dobble5.png')

lenM1 = M1.shape
# lenM2 = M2.shape
# lenM3 = M3.shape
# lenM4 = M4.shape
# lenM5 = M5.shape


def affiche1(M) :
    figure()
    axis('off')
    imshow(M)
    show()

def affiche2(M) :
    figure()
    axis('off')
    imshow(zone(M),cmap='binary')
    show()


def test_couleur(A) :
    a,b,c = A[:3]
    x=a+b+c
    if x > 2.5 :
        return False
    else :
        return True


def voisins(M,A) :
    a,b = A[:2]
    L=[[a,b]]
    lenM= M.shape
    if a > 0 :
        L.append([a-1,b])
    if b > 0 :
        L.append([a,b-1])
    if a < lenM[0]-1 :
        L.append([a+1,b])
    if b < lenM[1]-1 :
        L.append([a,b+1])
    return(L)
    # Commentaire de Léo : on pourrait ajouter les voisins en coin pour palier au problème de traits obliques (fantôme) qui disparaissent


# def zone(M) :
#     A= zeros_like(M)
#     lenA= A.shape
#     for i in range (lenA[0]) :
#         for j in range (lenA[1]) :
#             if test_couleur(M[i,j]) :
#                 A[i,j] = [0,1,0,1]
#             else :
#                 A[i,j] = [0,0,0,0]
#     return (A)


# def distance(Coord,centre) :
#     a,b = Coord
#     c,d = centre
#     return sqrt((a-c)**2+(b-d)**2)


# Commentaire de Léo : ceci est une proposition de la même fonction avec des noms de variables plus sympatiques comme exemple de noms de variables plus sympathiques, ça allait ici pour comprendre quand même mais ça peut être bon à prendre
def distance(O,point) :
    Ox,Oy = O
    x,y = point
    return sqrt((Ox-x)**2+(Oy-y)**2)


def disque(M) :
    i,j = M.shape[0],M.shape[1]
    centre = [i/2,j/2]
    A=M.copy()
    r=min(i//2,j//2) - min(i,j)/30
    for x in range (i) :
        for y in range (j) :
            if distance(centre, [x,y]) > r :
                A[x,y] = 0
    return A



def zone(M) :
    A= zeros_like(M)
    lenA= A.shape
    for i in range (lenA[0]) :
        for j in range (lenA[1]) :
            T=voisins(M,[i,j])
            test = 0
            for x in T :
                y,z=x
                if test_couleur(M[y,z]) :
                    test +=1
            if test>=4 :
                A[i,j] = 1
            else :
                A[i,j] = 0
    return (disque(A))



# def zone_leo(M) :
#     M = disque(M)
#     I = zeros_like(M)
#     p, q = I.shape
#     SEUIL_TEST = 4 # Pour éliminer les pixels "seuls" considérés comme un symbole, on les élimine s'ils ont moins de SEUIL_TEST voisins coloriées
#     for i in range (p) :
#         for j in range (q) :
#             if test_couleur(M[i,j]) :
#                 nb_test = 0
#                 for x,y in voisins(M,[i,j]) :
#                     if test_couleur(M[x,y]) :
#                         nb_test +=1
#                 if nb_test >= SEUIL_TEST :
#                     I[i,j] = 1
#                 else :
#                     I[i,j] = 0
#             else :
#                 I[i,j] = 0
#     return (I)




affiche2(M1)
# affiche2(M2)
# affiche2(M3)
# affiche2(M4)
# affiche2(M5)



M_import1 = zone(M1)
