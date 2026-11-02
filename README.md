# 🐍 Snake con Deep Q-Learning (DQN)

Agente que aprende a jugar al **Snake** por sí solo usando **Deep Q-Learning** con **PyTorch**. El juego está hecho desde cero con **Pygame**.

El agente empieza sin conocer las reglas. Recibe una recompensa al comer y un castigo al morir y, a base de prueba y error, una red neuronal aprende a estimar qué acción es mejor en cada situación.

Es la continuación de mi proyecto anterior, [Snake con Q-Learning tabular](https://github.com/PauCalvoJ/snake-q-learning), donde los valores Q se guardaban en una tabla.

<!-- Aquí puedes añadir un GIF del agente jugando: ![demo](demo.gif) -->

## 📈 Resultados

Evaluando el modelo sin exploración (`--jugar`):

| Versión | Puntuación media | Máxima |
|---|---|---|
| Q-Learning tabular (proyecto anterior) | ≈ 31 | 63–77 |
| DQN con estado de 16 valores | ≈ 34 | 70 |
| **DQN con espacio libre (20 valores)** | **≈ 90** | **151** |

La versión con espacio libre se entrenó desde cero unas 300 partidas (récord durante el entrenamiento: 165) y se evaluó en 26 partidas, así que su media es aproximada.

**Lo más interesante:** con el estado de 16 valores, la red neuronal no era mejor que la tabla. El motivo es que ese estado solo tiene unos cientos de combinaciones posibles: una tabla puede aprenderlas todas, así que la red no tiene nada que generalizar. Las dos llegaban a la misma estrategia ("no choques con lo de al lado y ve hacia la comida") y se quedaban en el mismo techo: cuando la serpiente es larga, se mete en un hueco cerrado por su propio cuerpo.

El salto llegó al darle al agente **cuánto espacio libre hay en cada dirección**. Con eso ve los callejones antes de entrar y la media casi se triplica. El límite no estaba en el algoritmo, sino en la información que recibía.

## 🚀 Uso

Requiere Python 3.10–3.12 (con versiones más nuevas puede que pygame o PyTorch aún no tengan paquete).

```bash
git clone https://github.com/PauCalvoJ/snake-deep-q-learning.git
cd snake-deep-q-learning
pip install -r requirements.txt
```

| Comando | Qué hace |
|---|---|
| `python main.py --jugar` | Muestra al modelo incluido jugando, sin entrenar |
| `python main.py --jugar --rapido --partidas 30` | Evalúa el modelo en 30 partidas y muestra la media |
| `python main.py` | Entrena viendo la ventana. Si existe `modelo_deep_snake.pth`, continúa desde él |
| `python main.py --rapido` | Entrena sin ventana, mucho más rápido |

Para entrenar desde cero, mueve o renombra `modelo_deep_snake.pth`. Ten en cuenta que cuanto mejor juega, más largas son las partidas: con más de 100 manzanas, una sola partida puede tardar varios segundos. Al entrenar se muestra el tiempo medio por paso (`ms/paso`).

## 🧠 Cómo funciona

### Estado (20 valores)
- **Peligro a 1 bloque** en cada dirección (arriba, abajo, izquierda, derecha): pared o cuerpo. Valor 0 o 1.
- **Peligro a 2 bloques** en cada dirección. 0 o 1.
- **Dirección actual** de la serpiente. 0 o 1.
- **Posición de la comida** respecto a la cabeza. 0 o 1.
- **Espacio libre** en cada dirección: qué fracción de las casillas libres del tablero podría alcanzar si se mueve hacia allí. Valor entre 0 y 1. Se calcula con una búsqueda en anchura (BFS) desde la casilla vecina; un valor cercano a 0 significa que es un callejón sin salida.

### Acciones
Moverse arriba, abajo, izquierda o derecha. Si la acción es dar media vuelta, se ignora y sigue recto.

### Recompensas
| Evento | Recompensa |
|---|---|
| Comer | +50 |
| Morir (chocar o pasar demasiado tiempo sin comer) | −100 |
| Cada paso (serpiente de menos de 25 bloques) | −0.05 |
| Cada paso (serpiente de 25 bloques o más) | −0.5 |

### Red neuronal
Red densa de **2 capas**: `20 entradas → 256 neuronas (ReLU) → 4 salidas`, una por acción. Cada salida es el valor Q estimado de esa acción.

### Entrenamiento
- **Q-learning:** el objetivo de cada paso es `r + γ · max Q(s', a')`, o solo `r` si la partida terminó.
- **Experience Replay:** los pasos se guardan en una memoria (hasta 100.000) y al final de cada partida se entrena con un lote aleatorio de 1000. Además se entrena con cada paso nada más darlo.
- **Red objetivo (target network):** los objetivos se calculan con una copia de la red que solo se actualiza cada 10 partidas. Así el objetivo no se mueve en cada paso y el entrenamiento es más estable.
- **Exploración ε-greedy:** al principio el agente elige acciones al azar con probabilidad 0.4, que baja hasta 0 en las primeras 80 partidas.
- **Rendimiento:** el estado se calcula una sola vez por paso y PyTorch usa un único hilo de CPU (`torch.set_num_threads(1)`), porque con una red tan pequeña repartir el trabajo entre núcleos cuesta más de lo que ahorra. Solo el cambio de hilos bajó el entrenamiento de unos 3,5 a 1,8 ms por paso.

| Hiperparámetro | Valor |
|---|---|
| Tasa de aprendizaje (Adam) | 0.001 |
| γ (descuento) | 0.9 |
| Tamaño del lote | 1000 |
| Actualización de la red objetivo | cada 10 partidas |

## 📁 Estructura

```
├── main.py                # Entrenar o jugar (argumentos de línea de comandos)
├── AgentDQL.py            # Agente: exploración, memoria y red objetivo
├── modelo.py              # Red neuronal (RedNeuronalQ) y paso de entrenamiento (EntrenadorQ)
├── SnakeGameDQL.py        # Juego, estado (incluido el espacio libre) y recompensas
└── modelo_deep_snake.pth  # Modelo ya entrenado (20 entradas)
```

## 📈 Evolución del proyecto

1. **[Q-Learning tabular](https://github.com/PauCalvoJ/snake-q-learning):** estado pequeño guardado en una tabla.
2. **Primera versión con red neuronal:** mismo estado de 8 valores que la versión tabular. Récord de 43.
3. **Estado de 16 valores:** añade peligro a 2 bloques y dirección, red objetivo y entrenamiento por lotes. Media ≈ 34, sin ventaja clara sobre la tabla.
4. **Espacio libre (versión actual):** media ≈ 90.

## 🔧 Posibles mejoras

- Entrenar aún más rápido, por ejemplo entrenando por lotes cada cierto número de pasos en vez de en todos.
- Darle a la red el tablero entero como imagen y usar una red convolucional, para que aprenda sola qué es importante en vez de usar características diseñadas a mano.
- Variantes como Double DQN o Dueling DQN.

## 📄 Licencia

[MIT](LICENSE)
