import random
from collections import deque

import numpy as np
import pygame

TAM_BLOQUE = 20
VELOCIDAD = 40

# Acciones: 0 arriba, 1 abajo, 2 izquierda, 3 derecha
DIRECCIONES = [[0, -TAM_BLOQUE], [0, TAM_BLOQUE], [-TAM_BLOQUE, 0], [TAM_BLOQUE, 0]]


class SnakeGame:
    def __init__(self, w=640, h=480, mostrar=True):
        self.w = w
        self.h = h
        self.mostrar = mostrar
        if self.mostrar:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("IA Snake - Deep Q-Learning")
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = [TAM_BLOQUE, 0]  # Empieza moviéndose a la derecha
        cabeza = [self.w // 2, self.h // 2]
        self.serpiente = [cabeza,
                          [cabeza[0] - TAM_BLOQUE, cabeza[1]],
                          [cabeza[0] - 2 * TAM_BLOQUE, cabeza[1]]]
        self.puntuacion = 0
        self._colocar_comida()
        self.interaccion_frame = 0
        self.done = False

    def _colocar_comida(self):
        while True:
            x = random.randint(0, (self.w - TAM_BLOQUE) // TAM_BLOQUE) * TAM_BLOQUE
            y = random.randint(0, (self.h - TAM_BLOQUE) // TAM_BLOQUE) * TAM_BLOQUE
            if [x, y] not in self.serpiente:
                self.comida = [x, y]
                return

    def get_state(self):
        """Estado de 16 valores (0 o 1) que ve el agente."""
        x, y = self.serpiente[0]
        b = TAM_BLOQUE

        estado = [
            # Peligro a 1 bloque: arriba, abajo, izquierda, derecha
            self.es_colision([x, y - b]), self.es_colision([x, y + b]),
            self.es_colision([x - b, y]), self.es_colision([x + b, y]),

            # Peligro a 2 bloques
            self.es_colision([x, y - 2 * b]), self.es_colision([x, y + 2 * b]),
            self.es_colision([x - 2 * b, y]), self.es_colision([x + 2 * b, y]),

            # Dirección actual
            self.direction == DIRECCIONES[0], self.direction == DIRECCIONES[1],
            self.direction == DIRECCIONES[2], self.direction == DIRECCIONES[3],

            # Dónde está la comida respecto a la cabeza
            self.comida[1] < y, self.comida[1] > y,
            self.comida[0] < x, self.comida[0] > x,

            # Espacio libre al que podría llegar moviéndose en cada dirección
            self._espacio_libre([x, y - b]), self._espacio_libre([x, y + b]),
            self._espacio_libre([x - b, y]), self._espacio_libre([x + b, y]),
        ]
        return np.array(estado, dtype=float)

    def _espacio_libre(self, inicio):
        """Fracción de las casillas libres que se pueden alcanzar desde 'inicio' (0 si está ocupada)."""
        if self.es_colision(inicio):
            return 0.0

        cuerpo = {tuple(p) for p in self.serpiente}
        visitadas = {tuple(inicio)}
        pendientes = deque([tuple(inicio)])
        while pendientes:
            x, y = pendientes.popleft()
            for dx, dy in DIRECCIONES:
                vecina = (x + dx, y + dy)
                if (vecina not in visitadas and vecina not in cuerpo and 0 <= vecina[0] < self.w and 0 <= vecina[1] < self.h ):
                    visitadas.add(vecina)
                    pendientes.append(vecina)

        libres = (self.w // TAM_BLOQUE) * (self.h // TAM_BLOQUE) - len(self.serpiente)
        return len(visitadas) / libres


    def es_colision(self, pt=None):
        if pt is None:
            pt = self.serpiente[0]
        # Pared
        if pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True
        # Su propio cuerpo
        return pt in self.serpiente[1:]

    def step(self, action):
        self._procesar_eventos()
        self.interaccion_frame += 1

        # No se permite dar media vuelta: se sigue en la misma dirección
        nueva_dir = DIRECCIONES[action]
        if nueva_dir != [-self.direction[0], -self.direction[1]]:
            self.direction = nueva_dir

        cabeza = self.serpiente[0]
        self.serpiente.insert(0, [cabeza[0] + self.direction[0], cabeza[1] + self.direction[1]])

        # Muere si choca o si lleva demasiado tiempo sin avanzar (evita bucles infinitos)
        if self.es_colision() or self.interaccion_frame > 100 * len(self.serpiente):
            self.done = True
            return self.get_state(), -100, self.done

        if self.serpiente[0] == self.comida:
            self.puntuacion += 1
            recompensa = 50
            self._colocar_comida()
        else:
            self.serpiente.pop()
            # Pequeño castigo por cada paso para que no se entretenga.
            # Es mayor cuando es larga para que busque la comida más rápido.
            recompensa = -0.05 if len(self.serpiente) < 25 else -0.5

        self._dibujar()
        return self.get_state(), recompensa, self.done

    def _procesar_eventos(self):
        """Atiende los eventos de la ventana para que no se congele y se pueda cerrar."""
        if not self.mostrar:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _dibujar(self):
        if not self.mostrar:
            return
        self.display.fill((0, 0, 0))
        for pt in self.serpiente:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt[0], pt[1], TAM_BLOQUE, TAM_BLOQUE))
        pygame.draw.rect(self.display, (255, 0, 0),
                         pygame.Rect(self.comida[0], self.comida[1], TAM_BLOQUE, TAM_BLOQUE))
        pygame.display.flip()
        self.clock.tick(VELOCIDAD)
