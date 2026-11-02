import random
from collections import deque

import torch

from modelo import EntrenadorQ, RedNeuronalQ

MAX_MEMORIA = 100_000
TAM_LOTE = 1000
LR = 0.001
GAMMA = 0.9
TAM_OCULTO = 256

# Exploración: epsilon baja en línea recta desde EPSILON_INICIAL hasta 0
# a lo largo de las primeras PARTIDAS_EXPLORACION partidas.
EPSILON_INICIAL = 0.4
PARTIDAS_EXPLORACION = 80

# Cada cuántas partidas se copian los pesos de la red principal a la red objetivo
ACTUALIZAR_OBJETIVO = 10


class AgentDQL:
    def __init__(self, tam_estado=20, num_acciones=4):
        self.num_acciones = num_acciones
        self.n_partidas = 0
        self.explorar = True
        self.memoria = deque(maxlen=MAX_MEMORIA)

        self.modelo = RedNeuronalQ(tam_estado, TAM_OCULTO, num_acciones)
        # Copia "congelada" de la red que se usa para calcular los objetivos.
        # Así el objetivo no cambia en cada paso y el entrenamiento es más estable.
        self.modelo_objetivo = RedNeuronalQ(tam_estado, TAM_OCULTO, num_acciones)
        self.actualizar_red_objetivo()

        self.entrenador = EntrenadorQ(self.modelo, self.modelo_objetivo, LR, GAMMA)

    def cargar(self, ruta):
        """Carga un modelo entrenado. Como ya sabe jugar, deja de explorar."""
        record = self.modelo.cargar(ruta)
        self.actualizar_red_objetivo()
        self.explorar = False
        return record

    def actualizar_red_objetivo(self):
        self.modelo_objetivo.load_state_dict(self.modelo.state_dict())

    def epsilon(self):
        if not self.explorar:
            return 0.0
        return max(0.0, EPSILON_INICIAL * (1 - self.n_partidas / PARTIDAS_EXPLORACION))

    def elegir_accion(self, estado):
        if random.random() < self.epsilon():
            return random.randrange(self.num_acciones)

        with torch.no_grad():
            q = self.modelo(torch.tensor(estado, dtype=torch.float))
        return int(torch.argmax(q).item())

    def recordar(self, estado, accion, recompensa, proximo_estado, done):
        self.memoria.append((estado, accion, recompensa, proximo_estado, done))

    def entrenar_memoria_corta(self, estado, accion, recompensa, proximo_estado, done):
        """Entrena solo con el último paso."""
        self.entrenador.paso_entrenamiento([estado], [accion], [recompensa], [proximo_estado], [done])

    def entrenar_memoria_larga(self):
        """Experience Replay: entrena con un lote aleatorio de pasos pasados."""
        if not self.memoria:
            return
        lote = random.sample(self.memoria, min(TAM_LOTE, len(self.memoria)))
        estados, acciones, recompensas, proximos_estados, dones = zip(*lote)
        self.entrenador.paso_entrenamiento(estados, acciones, recompensas, proximos_estados, dones)
