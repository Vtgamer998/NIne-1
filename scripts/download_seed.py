"""Script: gera corpus seed expandido com exemplos de codigo Python em PT-BR."""

import os

OUT_DIR = "nine/data/seed"
os.makedirs(OUT_DIR, exist_ok=True)


def write_seeds():
    examples = {}

    examples["seed_0"] = '''
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
'''

    examples["seed_1"] = '''
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
    vogais = set("aeiouAEIOUaeiouaeeeeeiouu")
    return sum(1 for c in s if c in vogais)
'''

    examples["seed_2"] = '''
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
'''

    examples["seed_3"] = '''
class Pilha:
    """Implementacao de uma pilha (LIFO) em Python."""

    def __init__(self):
        self.items = []

    def empilhar(self, item):
        self.items.append(item)

    def desempilhar(self):
        if self.vazia():
            return None
        return self.items.pop()

    def topo(self):
        if self.vazia():
            return None
        return self.items[-1]

    def vazia(self):
        return len(self.items) == 0

    def tamanho(self):
        return len(self.items)


class Fila:
    """Implementacao de uma fila (FIFO) em Python."""

    def __init__(self):
        self.items = []

    def enfileirar(self, item):
        self.items.append(item)

    def desenfileirar(self):
        if self.vazia():
            return None
        return self.items.pop(0)

    def vazia(self):
        return len(self.items) == 0

    def tamanho(self):
        return len(self.items)
'''

    examples["seed_4"] = '''
def busca_linear(lista, alvo):
    """Retorna o indice do alvo na lista, ou -1 se nao encontrado."""
    for i, item in enumerate(lista):
        if item == alvo:
            return i
    return -1


def busca_binaria(lista, alvo):
    """Retorna o indice do alvo em lista ordenada, ou -1."""
    esquerda, direita = 0, len(lista) - 1
    while esquerda <= direita:
        meio = (esquerda + direita) // 2
        if lista[meio] == alvo:
            return meio
        elif lista[meio] < alvo:
            esquerda = meio + 1
        else:
            direita = meio - 1
    return -1


def selection_sort(lista):
    """Ordena lista usando selection sort."""
    n = len(lista)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if lista[j] < lista[min_idx]:
                min_idx = j
        lista[i], lista[min_idx] = lista[min_idx], lista[i]
    return lista


def insertion_sort(lista):
    """Ordena lista usando insertion sort."""
    for i in range(1, len(lista)):
        chave = lista[i]
        j = i - 1
        while j >= 0 and lista[j] > chave:
            lista[j + 1] = lista[j]
            j -= 1
        lista[j + 1] = chave
    return lista
'''

    examples["seed_5"] = '''
def merge_sort(lista):
    """Ordena lista usando merge sort (dividir para conquistar)."""
    if len(lista) <= 1:
        return lista
    meio = len(lista) // 2
    esquerda = merge_sort(lista[:meio])
    direita = merge_sort(lista[meio:])
    return _merge(esquerda, direita)


def _merge(esquerda, direita):
    resultado = []
    i = j = 0
    while i < len(esquerda) and j < len(direita):
        if esquerda[i] <= direita[j]:
            resultado.append(esquerda[i])
            i += 1
        else:
            resultado.append(direita[j])
            j += 1
    resultado.extend(esquerda[i:])
    resultado.extend(direita[j:])
    return resultado


def quick_sort(lista):
    """Ordena lista usando quick sort."""
    if len(lista) <= 1:
        return lista
    pivo = lista[len(lista) // 2]
    menores = [x for x in lista if x < pivo]
    iguais = [x for x in lista if x == pivo]
    maiores = [x for x in lista if x > pivo]
    return quick_sort(menores) + iguais + quick_sort(maiores)
'''

    examples["seed_6"] = '''
class No:
    """No de uma arvore binaria de busca."""

    def __init__(self, valor):
        self.valor = valor
        self.esquerda = None
        self.direita = None


class ArvoreBinaria:
    """Arvore binaria de busca simples."""

    def __init__(self):
        self.raiz = None

    def inserir(self, valor):
        if self.raiz is None:
            self.raiz = No(valor)
        else:
            self._inserir(self.raiz, valor)

    def _inserir(self, no, valor):
        if valor < no.valor:
            if no.esquerda is None:
                no.esquerda = No(valor)
            else:
                self._inserir(no.esquerda, valor)
        else:
            if no.direita is None:
                no.direita = No(valor)
            else:
                self._inserir(no.direita, valor)

    def buscar(self, valor):
        return self._buscar(self.raiz, valor)

    def _buscar(self, no, valor):
        if no is None or no.valor == valor:
            return no
        if valor < no.valor:
            return self._buscar(no.esquerda, valor)
        return self._buscar(no.direita, valor)

    def em_ordem(self):
        resultado = []
        self._em_ordem(self.raiz, resultado)
        return resultado

    def _em_ordem(self, no, resultado):
        if no:
            self._em_ordem(no.esquerda, resultado)
            resultado.append(no.valor)
            self._em_ordem(no.direita, resultado)
'''

    examples["seed_7"] = '''
def contar_palavras(texto):
    """Retorna um dicionario com a frequencia de cada palavra."""
    palavras = texto.lower().split()
    frequencia = {}
    for palavra in palavras:
        palavra = palavra.strip(".,!?;:()[]{}")
        if palavra:
            frequencia[palavra] = frequencia.get(palavra, 0) + 1
    return frequencia


def top_n_palavras(texto, n=10):
    """Retorna as N palavras mais frequentes em um texto."""
    freq = contar_palavras(texto)
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]


def remover_duplicatas(lista):
    """Remove elementos duplicados mantendo a ordem original."""
    vistos = set()
    resultado = []
    for item in lista:
        if item not in vistos:
            vistos.add(item)
            resultado.append(item)
    return resultado
'''

    examples["seed_8"] = '''
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
'''

    examples["seed_9"] = '''
def ler_arquivo(caminho):
    """Le um arquivo de texto e retorna seu conteudo."""
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()


def escrever_arquivo(caminho, conteudo):
    """Escreve conteudo em um arquivo de texto."""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)


def listar_arquivos(diretorio, extensao=".py"):
    """Lista todos os arquivos com determinada extensao em um diretorio."""
    import os
    arquivos = []
    for root, _, files in os.walk(diretorio):
        for f in files:
            if f.endswith(extensao):
                arquivos.append(os.path.join(root, f))
    return arquivos
'''

    examples["seed_10"] = '''
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
'''

    examples["seed_11"] = '''
class ContaBancaria:
    """Classe simples simulando uma conta bancaria em PT-BR."""

    def __init__(self, titular, saldo=0):
        self.titular = titular
        self._saldo = saldo

    def depositar(self, valor):
        if valor > 0:
            self._saldo += valor
            return True
        return False

    def sacar(self, valor):
        if 0 < valor <= self._saldo:
            self._saldo -= valor
            return True
        return False

    def saldo(self):
        return self._saldo

    def __str__(self):
        return f"Conta de {self.titular}: R$ {self._saldo:.2f}"


class ContaPoupanca(ContaBancaria):
    """Conta poupanca com rendimento mensal."""

    def __init__(self, titular, saldo=0, taxa=0.005):
        super().__init__(titular, saldo)
        self.taxa = taxa

    def render(self):
        self._saldo += self._saldo * self.taxa
'''

    examples["seed_12"] = '''
import json


class Agenda:
    """Agenda de contatos simples."""

    def __init__(self):
        self.contatos = {}

    def adicionar(self, nome, telefone, email=""):
        self.contatos[nome] = {"telefone": telefone, "email": email}

    def remover(self, nome):
        return self.contatos.pop(nome, None)

    def buscar(self, nome):
        return self.contatos.get(nome)

    def listar(self):
        return list(self.contatos.keys())

    def salvar(self, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self.contatos, f, ensure_ascii=False, indent=2)

    def carregar(self, caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            self.contatos = json.load(f)
'''

    examples["seed_13"] = '''
def celsius_para_fahrenheit(c):
    return (c * 9/5) + 32


def fahrenheit_para_celsius(f):
    return (f - 32) * 5/9


def km_para_milhas(km):
    return km * 0.621371


def milhas_para_km(milhas):
    return milhas / 0.621371


def segundos_para_hms(segundos):
    """Converte segundos para horas, minutos e segundos."""
    h = segundos // 3600
    m = (segundos % 3600) // 60
    s = segundos % 60
    return h, m, s


def hms_para_segundos(h, m, s):
    return h * 3600 + m * 60 + s
'''

    examples["seed_14"] = '''
# tarefa: escreva uma funcao que calcula a media de uma lista
# solucao:

def media(lista):
    return sum(lista) / len(lista) if lista else 0


# tarefa: escreva uma funcao que calcula a mediana
# solucao:

def mediana(lista):
    if not lista:
        return 0
    ordenada = sorted(lista)
    n = len(ordenada)
    if n % 2 == 1:
        return ordenada[n // 2]
    return (ordenada[n // 2 - 1] + ordenada[n // 2]) / 2


# tarefa: escreva uma funcao que calcula o desvio padrao
# solucao:

import math

def desvio_padrao(lista):
    if len(lista) < 2:
        return 0
    m = media(lista)
    variancia = sum((x - m) ** 2 for x in lista) / (len(lista) - 1)
    return math.sqrt(variancia)
'''

    examples["seed_15"] = '''
def validar_cpf(cpf):
    """Validacao simples de CPF (11 digitos)."""
    cpf = "".join(c for c in cpf if c.isdigit())
    if len(cpf) != 11 or cpf == cpf[0] * 11:
        return False
    soma = sum(int(cpf[i]) * (10 - i) for i in range(9))
    dig1 = (soma * 10 % 11) % 10
    if dig1 != int(cpf[9]):
        return False
    soma = sum(int(cpf[i]) * (11 - i) for i in range(10))
    dig2 = (soma * 10 % 11) % 10
    return dig2 == int(cpf[10])


def validar_email(email):
    """Validacao simples de email."""
    import re
    padrao = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(padrao, email))
'''

    examples["seed_16"] = '''
def calcular_imc(peso, altura):
    """Calcula o Indice de Massa Corporal."""
    imc = peso / (altura ** 2)
    if imc < 18.5:
        classificacao = "Abaixo do peso"
    elif imc < 25:
        classificacao = "Peso normal"
    elif imc < 30:
        classificacao = "Sobrepeso"
    else:
        classificacao = "Obesidade"
    return imc, classificacao


def juros_compostos(principal, taxa, meses):
    """Calcula montante com juros compostos."""
    return principal * (1 + taxa) ** meses


def parcelamento(valor, parcelas, juros=0.02):
    """Calcula valor de parcelas com juros."""
    return valor * (juros * (1 + juros) ** parcelas) / ((1 + juros) ** parcelas - 1)
'''

    examples["seed_17"] = '''
class NoLista:
    def __init__(self, valor):
        self.valor = valor
        self.proximo = None


class ListaLigada:
    """Lista ligada simples."""

    def __init__(self):
        self.cabeca = None
        self._tamanho = 0

    def inserir(self, valor):
        novo = NoLista(valor)
        novo.proximo = self.cabeca
        self.cabeca = novo
        self._tamanho += 1

    def remover(self, valor):
        atual = self.cabeca
        anterior = None
        while atual and atual.valor != valor:
            anterior = atual
            atual = atual.proximo
        if atual is None:
            return False
        if anterior is None:
            self.cabeca = atual.proximo
        else:
            anterior.proximo = atual.proximo
        self._tamanho -= 1
        return True

    def buscar(self, valor):
        atual = self.cabeca
        while atual:
            if atual.valor == valor:
                return True
            atual = atual.proximo
        return False

    def tamanho(self):
        return self._tamanho

    def para_lista(self):
        resultado = []
        atual = self.cabeca
        while atual:
            resultado.append(atual.valor)
            atual = atual.proximo
        return resultado
'''

    examples["seed_18"] = '''
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
'''

    examples["seed_19"] = '''
import time

def decorator_temporizador(funcao):
    """Decorator que mede o tempo de execucao de uma funcao."""
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = funcao(*args, **kwargs)
        fim = time.time()
        print(f"{funcao.__name__} executou em {fim-inicio:.4f}s")
        return resultado
    return wrapper


@decorator_temporizador
def calcular_muito(num):
    total = 0
    for i in range(num):
        total += i ** 2
    return total
'''

    examples["seed_20"] = '''
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
'''

    examples["seed_21"] = '''
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
'''

    examples["seed_22"] = '''
def formatar_numero(n, casas=2):
    """Formata numero com separador de milhares."""
    return f"{n:,.{casas}f}"


def por_extenso(n):
    """Converte numero (0-99) para portugues."""
    unidades = ["zero", "um", "dois", "tres", "quatro", "cinco",
                "seis", "sete", "oito", "nove"]
    especiais = ["dez", "onze", "doze", "treze", "quatorze", "quinze",
                 "dezesseis", "dezessete", "dezoito", "dezenove"]
    dezenas = ["", "", "vinte", "trinta", "quarenta", "cinquenta",
               "sessenta", "setenta", "oitenta", "noventa"]
    if n < 10:
        return unidades[n]
    if n < 20:
        return especiais[n - 10]
    d = n // 10
    u = n % 10
    if u == 0:
        return dezenas[d]
    return f"{dezenas[d]} e {unidades[u]}"
'''

    examples["seed_23"] = '''
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
'''

    examples["seed_24"] = '''
import math

class Ponto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distancia(self, outro):
        return math.sqrt((self.x - outro.x)**2 + (self.y - outro.y)**2)

    def __add__(self, outro):
        return Ponto(self.x + outro.x, self.y + outro.y)

    def __str__(self):
        return f"Ponto({self.x}, {self.y})"


class Retangulo:
    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura

    def area(self):
        return self.largura * self.altura

    def perimetro(self):
        return 2 * (self.largura + self.altura)

    def __str__(self):
        return f"Retangulo({self.largura} x {self.altura})"
'''

    examples["seed_25"] = '''
from collections import Counter

def ordenar_por_frequencia(lista):
    """Ordena lista pelos elementos mais frequentes."""
    return [item for item, _ in Counter(lista).most_common()]


def intersecao(lista1, lista2):
    """Retorna a intersecao de duas listas."""
    return list(set(lista1) & set(lista2))


def uniao(lista1, lista2):
    """Retorna a uniao de duas listas sem duplicatas."""
    return list(set(lista1) | set(lista2))


def diferenca(lista1, lista2):
    """Retorna elementos que estao em lista1 mas nao em lista2."""
    return list(set(lista1) - set(lista2))
'''

    examples["seed_26"] = '''
from collections import Counter
import math

def calcular_estatisticas(lista):
    """Calcula media, mediana, moda, variancia e desvio padrao."""
    n = len(lista)
    if n == 0:
        return {}
    media = sum(lista) / n
    ordenada = sorted(lista)
    mediana = ordenada[n // 2] if n % 2 else (ordenada[n // 2 - 1] + ordenada[n // 2]) / 2
    freq = Counter(lista)
    max_freq = max(freq.values())
    moda = [k for k, v in freq.items() if v == max_freq]
    variancia = sum((x - media) ** 2 for x in lista) / (n - 1)
    desvio = math.sqrt(variancia)
    return {
        "media": media, "mediana": mediana, "moda": moda,
        "variancia": variancia, "desvio_padrao": desvio,
        "minimo": min(lista), "maximo": max(lista), "n": n
    }
'''

    examples["seed_27"] = '''
class Grafo:
    """Grafo simples usando lista de adjacencia."""

    def __init__(self, direcionado=False):
        self.adjacencias = {}
        self.direcionado = direcionado

    def adicionar_vertice(self, v):
        if v not in self.adjacencias:
            self.adjacencias[v] = []

    def adicionar_aresta(self, u, v):
        self.adicionar_vertice(u)
        self.adicionar_vertice(v)
        self.adjacencias[u].append(v)
        if not self.direcionado:
            self.adjacencias[v].append(u)

    def bfs(self, inicio):
        """Busca em largura (BFS)."""
        visitados = set()
        fila = [inicio]
        ordem = []
        while fila:
            v = fila.pop(0)
            if v not in visitados:
                visitados.add(v)
                ordem.append(v)
                fila.extend(self.adjacencias.get(v, []))
        return ordem

    def dfs(self, inicio):
        """Busca em profundidade (DFS)."""
        visitados = set()
        ordem = []

        def _dfs(v):
            visitados.add(v)
            ordem.append(v)
            for vizinho in self.adjacencias.get(v, []):
                if vizinho not in visitados:
                    _dfs(vizinho)

        _dfs(inicio)
        return ordem
'''

    examples["seed_28"] = '''
import re


def extrair_urls(texto):
    """Extrai URLs de um texto usando regex."""
    padrao = r"https?://[\\w./?=&%-]+"
    return re.findall(padrao, texto)


def extrair_emails(texto):
    """Extrai enderecos de email de um texto."""
    padrao = r"[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}"
    return re.findall(padrao, texto)


def limpar_texto(texto):
    """Remove pontuacao e normaliza espacos."""
    texto = re.sub(r"[^\\w\\s]", " ", texto)
    texto = re.sub(r"\\s+", " ", texto).strip()
    return texto.lower()


def dividir_em_frases(texto):
    """Divide texto em frases."""
    return re.split(r"[.!?]+", texto)
'''

    examples["seed_29"] = '''
import time
from contextlib import contextmanager


@contextmanager
def temporizador(nome="bloco"):
    """Context manager para medir tempo de execucao."""
    inicio = time.time()
    yield
    fim = time.time()
    print(f"{nome}: {fim-inicio:.4f}s")


class TabelaHash:
    """Implementacao simples de tabela hash."""
    def __init__(self, tamanho=100):
        self.tamanho = tamanho
        self.tabela = [[] for _ in range(tamanho)]

    def _hash(self, chave):
        return hash(chave) % self.tamanho

    def inserir(self, chave, valor):
        indice = self._hash(chave)
        for par in self.tabela[indice]:
            if par[0] == chave:
                par[1] = valor
                return
        self.tabela[indice].append([chave, valor])

    def buscar(self, chave):
        indice = self._hash(chave)
        for par in self.tabela[indice]:
            if par[0] == chave:
                return par[1]
        return None

    def remover(self, chave):
        indice = self._hash(chave)
        for i, par in enumerate(self.tabela[indice]):
            if par[0] == chave:
                self.tabela[indice].pop(i)
                return True
        return False
'''

    examples["seed_30"] = '''
# tarefa: escreva um decorator que valida tipos dos argumentos
# solucao:

def validar_tipos(**tipos):
    def decorator(funcao):
        def wrapper(*args, **kwargs):
            for (nome, valor), tipo_esperado in zip(kwargs.items(), tipos.values()):
                if not isinstance(valor, tipo_esperado):
                    raise TypeError(f"{nome} deve ser {tipo_esperado.__name__}")
            return funcao(*args, **kwargs)
        return wrapper
    return decorator


# tarefa: escreva uma funcao que aceita numero variavel de argumentos
# solucao:

def somar_tudo(*args, **kwargs):
    return sum(args) + sum(kwargs.values())


# tarefa: escreva um iterador personalizado
# solucao:

class Contador:
    def __init__(self, inicio=0, fim=10):
        self.atual = inicio
        self.fim = fim

    def __iter__(self):
        return self

    def __next__(self):
        if self.atual >= self.fim:
            raise StopIteration
        valor = self.atual
        self.atual += 1
        return valor
'''
    return examples


def generate_synthetic(target_chars=100000):
    """Generates synthetic Python examples to bulk up the corpus."""
    lines = []
    total = 0
    i = 0
    while total < target_chars:
        chunk = f"""
# Exemplo sintetico {i}
def func_exemplo_{i}(x):
    resultado = 0
    for j in range(x):
        resultado += j * {i + 1}
    return resultado


def processar_dados_{i}(dados):
    saida = []
    for item in dados:
        if item > {i % 50}:
            saida.append(item * {i + 2})
        else:
            saida.append(item // 2)
    return saida

"""
        lines.append(chunk)
        total += len(chunk)
        i += 1
    return "\n".join(lines)


def main():
    examples = write_seeds()
    for name, content in examples.items():
        path = os.path.join(OUT_DIR, f"{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Salvo {len(examples)} arquivos de exemplos em {OUT_DIR}")

    synthetic = generate_synthetic()
    synth_path = os.path.join(OUT_DIR, "synthetic_examples.py")
    with open(synth_path, "w", encoding="utf-8") as f:
        f.write(synthetic)
    print(f"Salvo ~{len(synthetic)} chars de exemplos sinteticos em {synth_path}")


if __name__ == "__main__":
    main()
