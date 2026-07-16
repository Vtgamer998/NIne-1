
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
