
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
