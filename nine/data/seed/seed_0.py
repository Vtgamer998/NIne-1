def fibonacci(n):
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
