from pylab import *
import os as os

os.chdir ('U:\mateo.brun')

M1 = imread('dobble1.png')
M2 = imread('dobble2.png')
M3 = imread('dobble3.png')
M4 = imread('dobble4.png')
M5 = imread('dobble5.png')

lenM1 = M1.shape
lenM2 = M2.shape
lenM3 = M3.shape
lenM4 = M4.shape
lenM5 = M5.shape


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


def distance(Coord,centre) :
    a,b = Coord
    c,d = centre
    return sqrt((a-c)**2+(b-d)**2)

def disque(M) :
    i,j = M.shape[0],M.shape[1]
    centre = [i/2,j/2]
    A=M.copy()
    r=min(i//2,j//2) - min(i,j)/30
    for a in range (i) :
        for b in range (j) :
            if distance([a,b],centre) > r :
                A[a,b] = 0
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


affiche2(M1)
# affiche2(M2)
# affiche2(M3)
# affiche2(M4)
# affiche2(M5)
