
def busca_linear(lista, alvo):
    """Retorna o indice do alvo na lista, ou -1 se nao encontrado."""
    for i, item in enumerate(lista):
        if item == alvo:
            return i
    return -1


def busca_binaria(lista, alvo):
    """Retorna o indice do alvo em lista ordenada, ou -1."""
    esquerda, direita = 0, len(lista) - 1
    while esquerda <= direita:
        meio = (esquerda + direita) // 2
        if lista[meio] == alvo:
            return meio
        elif lista[meio] < alvo:
            esquerda = meio + 1
        else:
            direita = meio - 1
    return -1


def selection_sort(lista):
    """Ordena lista usando selection sort."""
    n = len(lista)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if lista[j] < lista[min_idx]:
                min_idx = j
        lista[i], lista[min_idx] = lista[min_idx], lista[i]
    return lista


def insertion_sort(lista):
    """Ordena lista usando insertion sort."""
    for i in range(1, len(lista)):
        chave = lista[i]
        j = i - 1
        while j >= 0 and lista[j] > chave:
            lista[j + 1] = lista[j]
            j -= 1
        lista[j + 1] = chave
    return lista
