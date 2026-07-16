"""Gera dataset instrucional em PT-BR para fine-tuning LoRA do NINE-1."""

import json
import os

OUT_DIR = "nine/data"
os.makedirs(OUT_DIR, exist_ok=True)

INSTRUCTIONS = [
    # Fibonacci / matematica
    {"instruction": "escreva uma funcao fibonacci recursiva em python", "output": "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
    {"instruction": "escreva uma funcao fibonacci iterativa", "output": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
    {"instruction": "escreva uma funcao que calcula o fatorial de um numero", "output": "def fatorial(n):\n    if n <= 1:\n        return 1\n    return n * fatorial(n-1)"},
    {"instruction": "escreva uma funcao que verifica se um numero e primo", "output": "def primo(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"},
    {"instruction": "escreva uma funcao que calcula o MDC de dois numeros", "output": "def mdc(a, b):\n    while b:\n        a, b = b, a % b\n    return a"},
    {"instruction": "escreva uma funcao que calcula o MMC de dois numeros", "output": "def mmc(a, b):\n    return a * b // mdc(a, b)"},

    # Listas / arrays
    {"instruction": "escreva uma funcao que soma todos os elementos de uma lista", "output": "def soma_lista(lista):\n    return sum(lista)"},
    {"instruction": "escreva uma funcao que retorna o maior elemento de uma lista", "output": "def maior(lista):\n    return max(lista)"},
    {"instruction": "escreva uma funcao que inverte uma lista", "output": "def inverter(lista):\n    return lista[::-1]"},
    {"instruction": "escreva uma funcao que remove duplicatas de uma lista", "output": "def remover_duplicatas(lista):\n    return list(set(lista))"},
    {"instruction": "escreva uma funcao que conta vogais em uma string", "output": "def contar_vogais(texto):\n    vogais = set('aeiouAEIOU')\n    return sum(1 for c in texto if c in vogais)"},
    {"instruction": "escreva uma funcao que verifica se uma string e palindromo", "output": "def palindromo(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]"},

    # Ordenacao / busca
    {"instruction": "escreva bubble sort em python", "output": "def bubble_sort(lista):\n    n = len(lista)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if lista[j] > lista[j+1]:\n                lista[j], lista[j+1] = lista[j+1], lista[j]\n    return lista"},
    {"instruction": "escreva busca binaria em python", "output": "def busca_binaria(lista, alvo):\n    esq, dir = 0, len(lista) - 1\n    while esq <= dir:\n        meio = (esq + dir) // 2\n        if lista[meio] == alvo:\n            return meio\n        elif lista[meio] < alvo:\n            esq = meio + 1\n        else:\n            dir = meio - 1\n    return -1"},

    # Classes / OOP
    {"instruction": "crie uma classe Pilha em python", "output": "class Pilha:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        if not self.items:\n            return None\n        return self.items.pop()\n    def vazia(self):\n        return len(self.items) == 0"},
    {"instruction": "crie uma classe ContaBancaria em python", "output": "class ContaBancaria:\n    def __init__(self, titular, saldo=0):\n        self.titular = titular\n        self.saldo = saldo\n    def depositar(self, valor):\n        self.saldo += valor\n    def sacar(self, valor):\n        if valor <= self.saldo:\n            self.saldo -= valor\n            return True\n        return False"},

    # Utilitarios
    {"instruction": "escreva uma funcao que calcula a media de uma lista", "output": "def media(lista):\n    return sum(lista) / len(lista) if lista else 0"},
    {"instruction": "escreva uma funcao que converte celsius para fahrenheit", "output": "def celsius_para_fahrenheit(c):\n    return (c * 9/5) + 32"},
    {"instruction": "escreva uma funcao que valida email", "output": "import re\n\ndef email_valido(email):\n    padrao = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(padrao, email))"},
    {"instruction": "escreva uma funcao que ordena um dicionario por valor", "output": "def ordenar_por_valor(d):\n    return dict(sorted(d.items(), key=lambda x: x[1]))"},
    {"instruction": "escreva um gerador de numeros pares", "output": "def pares(n):\n    for i in range(n):\n        if i % 2 == 0:\n            yield i"},
    {"instruction": "escreva uma funcao que le um arquivo e retorna as linhas", "output": "def ler_arquivo(caminho):\n    with open(caminho, 'r', encoding='utf-8') as f:\n        return f.readlines()"},
]


def main():
    path = os.path.join(OUT_DIR, "instruct.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in INSTRUCTIONS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Salvo {len(INSTRUCTIONS)} exemplos instrucionais em {path}")
    total_chars = sum(len(item["instruction"]) + len(item["output"]) for item in INSTRUCTIONS)
    print(f"Total de caracteres: {total_chars}")


if __name__ == "__main__":
    main()
