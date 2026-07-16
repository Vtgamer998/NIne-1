
from functools import lru_cache


@lru_cache(maxsize=128)
def fibonacci_cache(n):
    """Fibonacci com memoizacao para desempenho O(n)."""
    if n < 2:
        return n
    return fibonacci_cache(n - 1) + fibonacci_cache(n - 2)


def knapsack(capacidade, pesos, valores, n):
    """Problema da mochila (knapsack) - programacao dinamica."""
    tabela = [[0] * (capacidade + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacidade + 1):
            if pesos[i - 1] <= w:
                tabela[i][w] = max(
                    valores[i - 1] + tabela[i - 1][w - pesos[i - 1]],
                    tabela[i - 1][w]
                )
            else:
                tabela[i][w] = tabela[i - 1][w]
    return tabela[n][capacidade]
