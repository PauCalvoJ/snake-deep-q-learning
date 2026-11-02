import argparse
import os

from AgentDQL import ACTUALIZAR_OBJETIVO, AgentDQL
from SnakeGameDQL import SnakeGame

RUTA_MODELO = "modelo_deep_snake.pth"


def entrenar(rapido, max_partidas):
    agente = AgentDQL()
    record = 0
    if os.path.exists(RUTA_MODELO):
        record = agente.cargar(RUTA_MODELO)
        print(f"Modelo cargado (récord {record}). Se continúa entrenando sin exploración.")
    else:
        print("No hay modelo previo. Empezando de cero...")

    juego = SnakeGame(mostrar=not rapido)

    while max_partidas == 0 or agente.n_partidas < max_partidas:
        estado = juego.get_state()
        accion = agente.elegir_accion(estado)
        nuevo_estado, recompensa, done = juego.step(accion)

        agente.entrenar_memoria_corta(estado, accion, recompensa, nuevo_estado, done)
        agente.recordar(estado, accion, recompensa, nuevo_estado, done)

        if done:
            agente.n_partidas += 1
            if juego.puntuacion > record:
                record = juego.puntuacion
                agente.modelo.guardar(RUTA_MODELO, record)

            print(f"Partida: {agente.n_partidas} | Puntuación: {juego.puntuacion} | Récord: {record}")

            agente.entrenar_memoria_larga()
            if agente.n_partidas % ACTUALIZAR_OBJETIVO == 0:
                agente.actualizar_red_objetivo()
            juego.reset()


def jugar(rapido, max_partidas):
    if not os.path.exists(RUTA_MODELO):
        print(f"No se encuentra {RUTA_MODELO}. Entrena primero con: python main.py")
        return

    agente = AgentDQL()
    agente.cargar(RUTA_MODELO)
    juego = SnakeGame(mostrar=not rapido)
    puntuaciones = []

    while max_partidas == 0 or len(puntuaciones) < max_partidas:
        _, _, done = juego.step(agente.elegir_accion(juego.get_state()))
        if done:
            puntuaciones.append(juego.puntuacion)
            media = sum(puntuaciones) / len(puntuaciones)
            print(f"Partida: {len(puntuaciones)} | Puntuación: {juego.puntuacion} | "
                  f"Media: {media:.1f} | Máxima: {max(puntuaciones)}")
            juego.reset()


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
