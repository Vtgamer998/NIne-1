
def buscar_em_matriz(matriz, alvo):
    """Busca um valor em uma matriz (retorna (linha, coluna) ou None)."""
    for i, linha in enumerate(matriz):
        for j, valor in enumerate(linha):
            if valor == alvo:
                return (i, j)
    return None


def somar_diagonais(matriz):
    """Soma as diagonais principal e secundaria de uma matriz quadrada."""
    n = len(matriz)
    diag_principal = sum(matriz[i][i] for i in range(n))
    diag_secundaria = sum(matriz[i][n - 1 - i] for i in range(n))
    return diag_principal, diag_secundaria


def rotacionar_matriz(matriz):
    """Rotaciona uma matriz 90 graus no sentido horario."""
    n = len(matriz)
    return [[matriz[n - 1 - j][i] for j in range(n)] for i in range(n)]
