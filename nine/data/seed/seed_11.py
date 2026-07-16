
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
