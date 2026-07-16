
def mdc(a, b):
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
