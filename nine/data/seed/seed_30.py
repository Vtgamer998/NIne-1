
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
