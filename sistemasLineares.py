#! /usr/bin/env python
#coding: utf-8

import numpy as np
import timeit
import copy
import os

def readFile(fileName):
  data = np.loadtxt(fileName)
  A = np.array(data[:-1])
  b = data[-1]
  return A,b


def substituicoesSucessivas(b,L):
  assert np.allclose(L, np.tril(L)), "Matriz L não é triangular inferior"
  n = L.shape[0]
  x = np.zeros(n, dtype=float)
  x[0] = b[0]/L[0,0]
  for i in range(1,n):
    soma = 0
    for j in range(0,i):
      soma = soma + L[i,j]*x[j]
    x[i] = (b[i]-soma)/L[i,i]
  return x


def substituicoesRetroativas(b,U):

  assert np.allclose(U, np.triu(U)), "Matriz L não é triangular superior"
  n = U.shape[0]
  x = np.zeros(n, dtype=float)
  x[-1] = b[-1]/U[-1,-1]
  for i in range(n-2,-1,-1):
    soma = 0
    for j in range(i+1,n):
      soma = soma + U[i,j]*x[j]
    x[i] = (b[i]-soma)/U[i,i]
  return x


def eliminacaoGauss(b, M):
  n = M.shape[0]
  if(np.allclose(M, np.tril(M))):
    x = substituicoesSucessivas(b,M)
    return b,M,np.zeros((n*n)/2, dtype=float),x
  elif (np.allclose(M, np.triu(M))):
    x = substituicoesRetroativas(b,M)
    return b,M,np.zeros((n*n)/2, dtype=float),x

  n = M.shape[0]
  k=0

  multiplicadores = np.zeros((n*n)/2, dtype=float)
  n_multiplicadores=0
  for k in range(0,n):
    if (M[k,k]!=0):
      pivo = M[k,k]
      l=0
      while(l+k<n-1):
         multiplicadores[n_multiplicadores] = M[k+1+l,k]/pivo
         n_multiplicadores = n_multiplicadores + 1
         linha_de_cima=M[k]
         j=0
         while (j<n):
           M[k+1+l,j]=M[k+1+l,j]-multiplicadores[n_multiplicadores-1]*linha_de_cima[j]
           j=j+1
         b[k+l+1]= b[k+l+1] - b[k]*multiplicadores[n_multiplicadores-1]
         l=l+1
    else:
         assert(M[k,k]!=0), "Não é possivel obter a solução por esse método."
  x = substituicoesRetroativas(b,M)

  return b, M, multiplicadores,x


def decomposicaoLU(b, M):

  b1 = copy.copy(b)
  M1 = copy.copy(M)

  if(np.allclose(M, np.tril(M))):
    x = substituicoesSucessivas(b,M)
    return x
  elif (np.allclose(M, np.triu(M))):
    x = substituicoesRetroativas(b,M)
    return x

  (b1,U,multiplicadores,xx) = eliminacaoGauss(b1,M1)
  n = len(b)
  L= np.zeros((n,n),dtype=float)
  preenchidos=0

  for i in range(0,n):
    for j in range (0,n):
      if (i==j): L[i][j]=1
      elif (i>j):
        L[i][j] = multiplicadores[preenchidos]
        preenchidos = preenchidos + 1
      else: L[i][j] =0

  y = substituicoesSucessivas(b,L)
  x = substituicoesRetroativas(y,U)

  return x


def criterioConvergencia_Linhas(M):
  n = M.shape[0]
  falha=0

  for i in range(n):
    soma=0
    for j in range(n):
      if (i!=j):soma = soma+ (M[i][j])
    if (soma/M[i][i]>1): falha =1
  if (falha==1): return False
  else: return True

def criterioSassenfeld(M):
  n = M.shape[0]
  soma=0
  b= np.ones(n,dtype=float)
  for i in range(1,n):
    soma = soma + abs(M[0][i])
  b[0]= soma/M[0][0]
  for j in range (0,n):
    somal=0
    for l in range (0,n):
     if (l!=j):
       somal= somal + (abs(M[j][l]))*b[l]
    b[j]=somal/M[j][j]
  if (max(b)<1): return True
  else: return False


def gaussJacobi(b,M):
  if ((criterioSassenfeld(M))==False):
    print("Aviso: Sistema não passa no teste de convergência de Sassenfeld.")
    resposta = input("Deseja continuar?<s/n>")
    if (resposta!='s'): return 0


  elif((criterioConvergencia_Linhas(M))==False):
    print("Aviso: Sistema não passa no teste de convergência das linhas.")
    resposta = input("Deseja continuar?<s/n>")
    if (resposta!='s'): return 0


  n= M.shape[0]
  x = np.zeros(n, dtype=float)
  xK = np.zeros(n, dtype=float)
  k = 0
  while((k<200)):
    for i in range (0,n):
      soma=0
      for j in range (0,n):
        if (i!=j): soma = soma + M[i][j]*x[j]
      xK[i]=(b[i]-soma)/M[i][i]

    x=list(xK)
    k = k +1
  return xK

def gaussSeidel(b,M):
  if ((criterioSassenfeld(M))==False):
      print("Aviso: Sistema não passa no teste de convergência de Sassenfeld.")
      resposta = input("Deseja continuar?<s/n>")
      if (resposta!='s'): return 0


  elif((criterioConvergencia_Linhas(M))==False):
     print("Aviso: Sistema não passa no teste de convergência das linhas.")
     resposta = input("Deseja continuar?<s/n>")
     if (resposta!='s'): return 0

  n= M.shape[0]
  x = np.zeros(n, dtype=float)
  xK = np.zeros(n, dtype=float)

  for k in range(0,100):
    for i in range (0,n):
      soma=0
      for j in range (0,n):
        if (j!=i): soma = soma + M[i][j]*xK[j]
      xK[i]=(b[i]-soma)/M[i][i]

  return xK

def residuo(b, L, x):
  n= L.shape[0]
  y = np.zeros(n, dtype=float)
  valorResiduo = np.zeros(n, dtype=float)
  for i in range(0,n):
    soma=0
    for j in range(0, n):
      soma = soma + L[i][j] * x[j]
    y[i]= soma
  for k in range (0, n):
      valorResiduo[k] = abs(y[k]-b[k])
  return valorResiduo



L= np.zeros((4,4),dtype=float)
(L,b) = readFile('matriz.txt')
mL=copy.copy(L)
mB= copy.copy(b)


print("\n########## ELIMINACAO DE GAUSS ##########n")

ini = timeit.timeit()
(L,b) = readFile('matriz.txt')
mL=copy.copy(L)
mb= copy.copy(b)
(b1,m1,mul,x) = eliminacaoGauss(b,L)
fim = timeit.timeit()
print("A solução usando este método é:\n " + str(x))
r = residuo(mb, mL, x)
print("\nResiduo:" + str(r))
print("\nTempo de execução: " + str(fim-ini))

print("\n########## DECOMPOSIÇÃO LU ##########n")
ini = timeit.timeit()
(L,b) = readFile('matriz.txt')
mL=copy.copy(L)
mb= copy.copy(b)
x = decomposicaoLU(b,L)
fim = timeit.timeit()
print("A solução usando este método é:\n " + str(x))
r = residuo(mb, mL, x)
print("Residuo: " + str(r))
print("\nTempo de execução: " + str(fim-ini))

print("\n########## GAUSS-JACOBI ##########\n")
ini = timeit.timeit()
(L,b) = readFile('matriz.txt')
mL=copy.copy(L)
mb= copy.copy(b)
x = gaussJacobi(b,L)
fim = timeit.timeit()
print("A solução usando este método é:\n " + str(x))
r = residuo(mb, mL, x)
print("Residuo: ",r)
print("\nTempo de execução: " + str(fim-ini))

print("\n########## GAUSS-SEIDEL ########## \n")
ini = timeit.timeit()
(L,b) = readFile('matriz.txt')
mL=copy.copy(L)
mb= copy.copy(b)
x = gaussSeidel(b,L)
fim = timeit.timeit()
print("A solução usando este método é:\n " + str(x))
r = residuo(mb, mL, x)
print("Residuo: ",r)
print("\nTempo de execução: " + str(fim-ini))

print("\n\nDigite outra matriz no arquivo 'matriz.txt' para outros cálculos.")
