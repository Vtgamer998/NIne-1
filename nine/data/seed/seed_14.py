
# tarefa: escreva uma funcao que calcula a media de uma lista
# solucao:

def media(lista):
    return sum(lista) / len(lista) if lista else 0


# tarefa: escreva uma funcao que calcula a mediana
# solucao:

def mediana(lista):
    if not lista:
        return 0
    ordenada = sorted(lista)
    n = len(ordenada)
    if n % 2 == 1:
        return ordenada[n // 2]
    return (ordenada[n // 2 - 1] + ordenada[n // 2]) / 2


# tarefa: escreva uma funcao que calcula o desvio padrao
# solucao:

import math

def desvio_padrao(lista):
    if len(lista) < 2:
        return 0
    m = media(lista)
    variancia = sum((x - m) ** 2 for x in lista) / (len(lista) - 1)
    return math.sqrt(variancia)
