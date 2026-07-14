"""Script exemplo: baixa um subset leve de 'Tiny Codes' (jogos em PT-BR) e prepara corpus.
Como esta dependencia de internet, e opcional: usado apenas se o usuario quiser.
"""

import os
import urllib.request

SAMPLE_URLS = [
    # URLs de exemplo (substituivel) — coloque arquivos .py aqui
]

OUT_DIR = "nine/data/seed"
os.makedirs(OUT_DIR, exist_ok=True)


SAMPLES = [
    '''def fibonacci(n):
    """Retorna o n-esimo termo de Fibonacci."""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


def fatorial(n):
    """Retorna n! recursivamente."""
    return 1 if n <= 1 else n * fatorial(n-1)


def primo(n):
    """Verifica se n e primo."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def bubble_sort(lista):
    """Ordena uma lista em ordem crescente usando bubble sort."""
    lista = list(lista)
    n = len(lista)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lista[j] > lista[j + 1]:
                lista[j], lista[j + 1] = lista[j + 1], lista[j]
    return lista


if __name__ == "__main__":
    print("fibonacci(10) =", fibonacci(10))
    print("fatorial(6)   =", fatorial(6))
    print("primo(29)     =", primo(29))
    print("bubble_sort([5,1,4,1,2])=", bubble_sort([5, 1, 4, 1, 2]))
''',
    '''# tarefa: escreva uma funcao que soma todos os pares de uma lista
# solucao:

def soma_pares(lista):
    """Retorna a soma de todos os numeros pares de uma lista."""
    return sum(x for x in lista if x % 2 == 0)


# tarefa: escreva uma funcao que inverte uma string
# solucao:

def inverte(s):
    """Retorna a string invertida."""
    return s[::-1]


# tarefa: escreva uma funcao que conta vogais
# solucao:

def conta_vogais(s):
    """Conta vogais (pt-br) em uma string."""
    vogais = set("aeiouAEIOUáéíóúÁÉÍÓÚãõÃÕ")
    return sum(1 for c in s if c in vogais)
''',
    '''def mdc(a, b):
    while b:
        a, b = b, a % b
    return a


def mmc(a, b):
    return a * b // mdc(a, b)


def eh_palindromo(s: str) -> bool:
    s = "".join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]


def anagramas(s1: str, s2: str) -> bool:
    return sorted(s1.replace(" ", "").lower()) == sorted(s2.replace(" ", "").lower())


def fibonacci_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
]


def main():
    for i, s in enumerate(SAMPLES):
        path = os.path.join(OUT_DIR, f"seed_{i}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    print(f"Salvo {len(SAMPLES)} arquivos em {OUT_DIR}")


if __name__ == "__main__":
    main()
