import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RedNeuronalQ(nn.Module):
    """Red densa de 2 capas: estado -> capa oculta (ReLU) -> un valor Q por acción."""

    def __init__(self, tam_entrada, tam_oculto, tam_salida):
        super().__init__()
        self.capa1 = nn.Linear(tam_entrada, tam_oculto)
        self.capa2 = nn.Linear(tam_oculto, tam_salida)

    def forward(self, x):
        x = F.relu(self.capa1(x))
        return self.capa2(x)

    def guardar(self, ruta, record):
        torch.save({"pesos": self.state_dict(), "record": record}, ruta)

    def cargar(self, ruta):
        """Carga los pesos y devuelve el récord guardado (0 si el archivo no lo tiene)."""
        datos = torch.load(ruta, weights_only=True)
        if "pesos" in datos:
            self.load_state_dict(datos["pesos"])
            return datos["record"]
        # Formato antiguo: el archivo solo contiene los pesos
        self.load_state_dict(datos)
        return 0


class EntrenadorQ:
    def __init__(self, modelo, modelo_objetivo, lr, gamma):
        self.modelo = modelo
        self.modelo_objetivo = modelo_objetivo
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
        self.criterio = nn.MSELoss()

    def paso_entrenamiento(self, estados, acciones, recompensas, proximos_estados, dones):
        estados = torch.tensor(np.array(estados), dtype=torch.float)
        proximos_estados = torch.tensor(np.array(proximos_estados), dtype=torch.float)
        acciones = torch.tensor(acciones, dtype=torch.long)
        recompensas = torch.tensor(recompensas, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        # Q(s, a) que predice la red para las acciones que se tomaron
        q_actual = self.modelo(estados).gather(1, acciones.unsqueeze(1)).squeeze(1)

        # Objetivo de Q-learning: r + gamma * max Q_objetivo(s').
        # Si la partida terminó no hay futuro, así que el objetivo es solo r.
        with torch.no_grad():
            q_siguiente = self.modelo_objetivo(proximos_estados).max(dim=1).values
        q_objetivo = recompensas + self.gamma * q_siguiente * (1 - dones)

        perdida = self.criterio(q_actual, q_objetivo)
        self.optimizer.zero_grad()
        perdida.backward()
        self.optimizer.step()
        return perdida.item()
