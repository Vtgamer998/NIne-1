
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
