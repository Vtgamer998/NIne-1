
# tarefa: crie um gerador de numeros pares
# solucao:

def gerar_pares(n):
    for i in range(n):
        if i % 2 == 0:
            yield i


# tarefa: crie um gerador para ler arquivos grandes linha por linha
# solucao:

def ler_linhas(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            yield linha.strip()


# tarefa: use compressao de lista para filtrar numeros
# solucao:

def numeros_divisiveis(lista, divisor):
    return [n for n in lista if n % divisor == 0]


def dicionario_quadrados(n):
    return {x: x**2 for x in range(n)}


def pares_ordenados(lista1, lista2):
    return [(a, b) for a in lista1 for b in lista2]
