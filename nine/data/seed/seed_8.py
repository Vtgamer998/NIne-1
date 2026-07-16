
def matriz_transpor(matriz):
    """Retorna a transposta de uma matriz."""
    return [[matriz[j][i] for j in range(len(matriz))]
            for i in range(len(matriz[0]))]


def matriz_multiplicar(A, B):
    """Multiplica duas matrizes."""
    linhas_A, colunas_A = len(A), len(A[0])
    linhas_B, colunas_B = len(B), len(B[0])
    if colunas_A != linhas_B:
        raise ValueError("Dimensoes incompativeis")
    resultado = [[0] * colunas_B for _ in range(linhas_A)]
    for i in range(linhas_A):
        for j in range(colunas_B):
            for k in range(colunas_A):
                resultado[i][j] += A[i][k] * B[k][j]
    return resultado


def matriz_identidade(n):
    """Retorna uma matriz identidade n x n."""
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
