# tarefa: escreva uma funcao que soma todos os pares de uma lista
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
