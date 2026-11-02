import argparse
import os
import time
import torch

from AgentDQL import ACTUALIZAR_OBJETIVO, AgentDQL
from SnakeGameDQL import SnakeGame

RUTA_MODELO = "modelo_deep_snake.pth"

# Con una red tan pequeña, usar un solo hilo de CPU suele ser más rápido que repartir el trabajo
torch.set_num_threads(1)

def entrenar(rapido, max_partidas):
    agente = AgentDQL()
    record = 0
    if os.path.exists(RUTA_MODELO):
        record = agente.cargar(RUTA_MODELO)
        print(f"Modelo cargado (récord {record}). Se continúa entrenando sin exploración.")
    else:
        print("No hay modelo previo. Empezando de cero...")

    juego = SnakeGame(mostrar=not rapido)
    estado = juego.get_state()
    pasos = 0
    inicio = time.time()

    while max_partidas == 0 or agente.n_partidas < max_partidas:
        accion = agente.elegir_accion(estado)
        nuevo_estado, recompensa, done = juego.step(accion)
        pasos += 1

        agente.entrenar_memoria_corta(estado, accion, recompensa, nuevo_estado, done)
        agente.recordar(estado, accion, recompensa, nuevo_estado, done)

        # El estado nuevo es el de partida del siguiente paso: no hace falta recalcularlo
        estado = nuevo_estado

        if done:
            agente.n_partidas += 1
            if juego.puntuacion > record:
                record = juego.puntuacion
                agente.modelo.guardar(RUTA_MODELO, record)

            ms_por_paso = (time.time() - inicio) / pasos * 1000
            print(f"Partida: {agente.n_partidas} | Puntuación: {juego.puntuacion} | "
                  f"Récord: {record} | {ms_por_paso:.2f} ms/paso")

            agente.entrenar_memoria_larga()
            if agente.n_partidas % ACTUALIZAR_OBJETIVO == 0:
                agente.actualizar_red_objetivo()

            juego.reset()
            estado = juego.get_state()
            pasos = 0
            inicio = time.time()


def jugar(rapido, max_partidas):
    if not os.path.exists(RUTA_MODELO):
        print(f"No se encuentra {RUTA_MODELO}. Entrena primero con: python main.py")
        return

    agente = AgentDQL()
    agente.cargar(RUTA_MODELO)
    juego = SnakeGame(mostrar=not rapido)
    puntuaciones = []

    estado = juego.get_state()
    while max_partidas == 0 or len(puntuaciones) < max_partidas:
        estado, _, done = juego.step(agente.elegir_accion(estado))
        if done:
            puntuaciones.append(juego.puntuacion)
            media = sum(puntuaciones) / len(puntuaciones)
            print(f"Partida: {len(puntuaciones)} | Puntuación: {juego.puntuacion} | "
                  f"Media: {media:.1f} | Máxima: {max(puntuaciones)}")
            juego.reset()
            estado = juego.get_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake con Deep Q-Learning")
    parser.add_argument("--jugar", action="store_true",
                        help="ver jugar al modelo guardado, sin entrenar")
    parser.add_argument("--rapido", action="store_true",
                        help="sin ventana: entrena o evalúa mucho más rápido")
    parser.add_argument("--partidas", type=int, default=0,
                        help="número de partidas (0 = hasta cerrar la ventana o Ctrl+C)")
    args = parser.parse_args()

    try:
        if args.jugar:
            jugar(args.rapido, args.partidas)
        else:
            entrenar(args.rapido, args.partidas)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
