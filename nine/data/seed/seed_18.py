
import random
import string

def gerar_senha(tamanho=12):
    """Gera uma senha aleatoria segura."""
    caracteres = string.ascii_letters + string.digits + "!@#$%&*"
    return "".join(random.choice(caracteres) for _ in range(tamanho))


def cifra_cesar(texto, deslocamento):
    """Cifra/decifra texto com cifra de Cesar."""
    resultado = []
    for c in texto:
        if c.isalpha():
            maiuscula = c.isupper()
            base = ord("A") if maiuscula else ord("a")
            novo = chr((ord(c) - base + deslocamento) % 26 + base)
            resultado.append(novo)
        else:
            resultado.append(c)
    return "".join(resultado)


def hash_simples(texto):
    """Hash simples de string (djb2)."""
    h = 5381
    for c in texto:
        h = (h * 33 + ord(c)) & 0xFFFFFFFF
    return h
