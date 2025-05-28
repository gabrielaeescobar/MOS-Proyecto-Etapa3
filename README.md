# Proyecto Etapa 3 - Metaheurísticas para CVRP

## Descripción del método

En esta fase se implementó un **Algoritmo Genético (GA)** como método metaheurístico para resolver el problema de ruteo de vehículos con capacidad (CVRP). El algoritmo utiliza:

- Representación de soluciones como listas de rutas (una por vehículo).
- Operadores personalizados de cruce y mutación para generar nuevas soluciones.
- Heurística de reparación para asegurar que las soluciones sean factibles (respeto por la capacidad y visitas únicas).
- Evaluación basada en la distancia total recorrida por los vehículos.

## Reproducción de los experimentos

Las instancias utilizadas fueron:

- `Proyecto_Caso_Base`
- `Proyecto_C_Caso2`
- `Proyecto_C_Caso3`

Cada carpeta contiene los siguientes archivos obligatorios:

- `clients.csv`: Lista de clientes con sus demandas.
- `vehicles.csv`: Definición de la flota (se omite la columna de capacidad en esta fase).
- `depots.csv`: Se utiliza únicamente el depósito definido en `Proyecto_Caso_Base`.

El notebook principal para ejecutar el algoritmo es:

MOS-Proyecto-Etapa3_casoBaseGA.ipynb


Desde allí se realiza todo el pipeline: carga de datos, ejecución del algoritmo genético, visualización de resultados, verificación y análisis de escalabilidad.

## Parámetros del algoritmo y reproducibilidad

El algoritmo se ejecutó con **3 semillas diferentes por cada caso** para observar la variabilidad y robustez del método. Las semillas se pueden configurar en el código, y los resultados se almacenan por separado para cada repetición.

Se registraron los siguientes indicadores por cada repetición:

- Tiempo de ejecución
- Fitness (distancia total)
- Porcentaje de demanda cubierta
- Curva de convergencia

## Archivos generados

Por cada instancia, el algoritmo produce un archivo `.csv` de verificación con el detalle de las rutas generadas:

- `verificacion_metaheuristica_GA_Caso_Base.csv`
- `verificacion_metaheuristica_GA_Caso_2.csv`
- `verificacion_metaheuristica_GA_Caso_3.csv`

Y archivos `.html` con las rutas visualizadas en el mapa:

- `mapa_rutas_caso_base.html`
- `mapa_rutas_caso_2.html`
- `mapa_rutas_caso_3.html`

## Visualizaciones incluidas

El notebook genera los siguientes gráficos:

- Curva de convergencia del fitness para cada caso.
- Gráficos comparativos del tiempo de ejecución y demanda cubierta.
- Mapas interactivos de rutas generadas (HTML).
- Diagramas de caja con las distancias recorridas por vehículo.
- Comparación final de desempeño del algoritmo en los tres casos (fitness, escalabilidad, cobertura).


