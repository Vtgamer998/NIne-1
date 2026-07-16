
def numeros_primos_ate(n):
    """Retorna lista de numeros primos ate n usando crivo de Eratostenes."""
    if n < 2:
        return []
    crivo = [True] * (n + 1)
    crivo[0] = crivo[1] = False
    for i in range(2, int(n**0.5) + 1):
        if crivo[i]:
            for j in range(i * i, n + 1, i):
                crivo[j] = False
    return [i for i, primo in enumerate(crivo) if primo]


def fatorar(n):
    """Retorna os fatores primos de n."""
    fatores = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            fatores.append(d)
            n //= d
        d += 1
    if n > 1:
        fatores.append(n)
    return fatores


def fibonacci_lista(n):
    """Retorna os primeiros n numeros de Fibonacci."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib
