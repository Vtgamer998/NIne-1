
from collections import Counter

def ordenar_por_frequencia(lista):
    """Ordena lista pelos elementos mais frequentes."""
    return [item for item, _ in Counter(lista).most_common()]


def intersecao(lista1, lista2):
    """Retorna a intersecao de duas listas."""
    return list(set(lista1) & set(lista2))


def uniao(lista1, lista2):
    """Retorna a uniao de duas listas sem duplicatas."""
    return list(set(lista1) | set(lista2))


def diferenca(lista1, lista2):
    """Retorna elementos que estao em lista1 mas nao em lista2."""
    return list(set(lista1) - set(lista2))
