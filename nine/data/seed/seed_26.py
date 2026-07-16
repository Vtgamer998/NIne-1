
from collections import Counter
import math

def calcular_estatisticas(lista):
    """Calcula media, mediana, moda, variancia e desvio padrao."""
    n = len(lista)
    if n == 0:
        return {}
    media = sum(lista) / n
    ordenada = sorted(lista)
    mediana = ordenada[n // 2] if n % 2 else (ordenada[n // 2 - 1] + ordenada[n // 2]) / 2
    freq = Counter(lista)
    max_freq = max(freq.values())
    moda = [k for k, v in freq.items() if v == max_freq]
    variancia = sum((x - media) ** 2 for x in lista) / (n - 1)
    desvio = math.sqrt(variancia)
    return {
        "media": media, "mediana": mediana, "moda": moda,
        "variancia": variancia, "desvio_padrao": desvio,
        "minimo": min(lista), "maximo": max(lista), "n": n
    }
