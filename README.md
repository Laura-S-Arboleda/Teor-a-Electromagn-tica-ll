
<div align="center">

# ⚡ Simulaciones Computacionales de las Ecuaciones de Maxwell

### Teoría Electromagnética II · Universidad del Tolima

<br>

<img src="https://raw.githubusercontent.com/Laura-S-Arboleda/Teor-a-Electromagn-tica-ll/main/facu.png" width="120">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/Laura-S-Arboleda/Teor-a-Electromagn-tica-ll/main/fisica_logo_3.png" width="120">

<br><br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Electromagnetism](https://img.shields.io/badge/Electromagnetism-⚡-8A2BE2?style=for-the-badge)
![Maxwell](https://img.shields.io/badge/Maxwell's_Equations-🧲-F5A623?style=for-the-badge)
![Universidad del Tolima](https://img.shields.io/badge/Universidad_del_Tolima-Physics-006B3C?style=for-the-badge)

<br>

**Del modelo matemático a la simulación computacional.**

</div>

---

# 🌌 Sobre el proyecto

Las **ecuaciones de Maxwell** constituyen uno de los pilares fundamentales
del electromagnetismo. En este proyecto se estudian diferentes leyes de
Maxwell mediante **simulaciones computacionales**, métodos numéricos y
visualizaciones gráficas.

El objetivo es conectar la formulación matemática con una representación
computacional que permita **visualizar, analizar y verificar** el
comportamiento de los campos electromagnéticos.

---

# 🧪 Simulaciones

El repositorio contiene tres simulaciones principales:

| | Simulación | Ley de Maxwell | Tema |
|:---:|---|---|---|
| 🧲 | **Faraday Moving** | Ley de Faraday | Inducción electromagnética |
| ☁️ | **Gaussian Cloud** | Ley de Gauss | Campo eléctrico de una distribución de carga |
| 🧭 | **Gauss's Law for Magnetism** | Ley de Gauss para el magnetismo | Divergencia y campo magnético |

---

# 🚀 Puesta en marcha

## 📥 1. Clonar el repositorio

```bash
git clone https://github.com/Laura-S-Arboleda/Teor-a-Electromagn-tica-ll.git
````

Entrar al directorio:

```bash
cd Teor-a-Electromagn-tica-ll
```

---

## 🐍 2. Crear un entorno virtual

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

# ⚡ 01 · Faraday Moving

## Ley de Faraday

La primera simulación estudia la **inducción electromagnética** asociada
con la Ley de Faraday.

La simulación cuenta con un archivo `main.py` que inicia directamente
la aplicación.

### ▶️ Ejecución

```bash
cd faraday_moving
pip install -r requirements.txt
python main.py
```

No es necesario utilizar argumentos adicionales.

---

# ☁️ 02 · Gaussian Cloud

## Ley de Gauss

Esta simulación representa una **distribución esférica de carga** y permite
estudiar el campo eléctrico y el flujo eléctrico mediante la Ley de Gauss.

### ▶️ Ejecución

```bash
cd gaussian_cloud
pip install -r requirements.txt
python main.py
```

La simulación se inicia directamente mediante `main.py`.

---

# 🧲 03 · Gauss's Law for Magnetism

## Ley de Gauss para el magnetismo

La tercera simulación estudia la condición:

$$
\nabla \cdot \vec{B}=0
$$

Esta simulación tiene una estructura diferente a las anteriores.

El archivo `main.py` utiliza **argumentos de línea de comandos** para
seleccionar el análisis que se desea ejecutar.

Por esta razón:

```bash
python main.py
```

**NO ejecuta todos los análisis.**

Al no especificar ningún argumento, el programa utiliza por defecto
el modo `validation`.

---

# 🎛️ Modos de ejecución

La simulación dispone de los siguientes modos:

```text
validation
divergence
convergence
visualization
animation
```

---

## 🔬 Validation

Compara el campo magnético obtenido numéricamente con la solución
analítica sobre el eje y calcula el error relativo.

```bash
python main.py --mode validation
```

También puede ejecutarse simplemente:

```bash
python main.py
```

ya que `validation` es el modo predeterminado.

---

## 📐 Divergence

Permite analizar numéricamente la divergencia del campo magnético:

$$
\nabla \cdot \vec{B}=0
$$

Ejecutar:

```bash
python main.py --mode divergence
```

---

## 📊 Convergence

Realiza un análisis de convergencia del método numérico utilizado.

```bash
python main.py --mode convergence
```

Los resultados del análisis se muestran en la terminal.

---

## 🖼️ Visualization

Ejecuta la visualización del campo magnético.

```bash
python main.py --mode visualization
```

---

## 🎥 Animation

La simulación permite generar animaciones en **2D** y **3D**.

### Animación 2D

```bash
python main.py --mode animation --anim-mode 2d
```

### Animación 3D

```bash
python main.py --mode animation --anim-mode 3d
```

Si no se especifica `--anim-mode`, la animación utiliza **3D por defecto**.

---

# 🕹️ Comandos rápidos

Si solo necesitas saber qué ejecutar:

| Simulación                | Comando                                          |
| ------------------------- | ------------------------------------------------ |
| 🧲 Faraday Moving         | `python main.py`                                 |
| ☁️ Gaussian Cloud         | `python main.py`                                 |
| 🧭 Gauss — Validation     | `python main.py --mode validation`               |
| ∇ Gauss — Divergence      | `python main.py --mode divergence`               |
| 📊 Gauss — Convergence    | `python main.py --mode convergence`              |
| 🖼️ Gauss — Visualization | `python main.py --mode visualization`            |
| 🎥 Gauss — Animation 2D   | `python main.py --mode animation --anim-mode 2d` |
| 🌐 Gauss — Animation 3D   | `python main.py --mode animation --anim-mode 3d` |

---

# 🧠 ¿Por qué Gauss magnético tiene varios modos?

La simulación de Gauss para el magnetismo fue diseñada como una aplicación
modular que permite estudiar diferentes aspectos del modelo.

El argumento:

```bash
--mode
```

determina qué análisis se ejecuta.

La estructura puede visualizarse de la siguiente manera:

```text
                         🧲 MAIN.PY
                             │
                             ▼
                         --mode
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
     validation         divergence        convergence
          │                  │                  │
          ▼                  ▼                  ▼
      🔬 Validación        ∇ · B = 0        📊 Error
          │
          │
          └──────────────────┬──────────────────┐
                             │                  │
                             ▼                  ▼
                       visualization        animation
                                                │
                                          ┌─────┴─────┐
                                          ▼           ▼
                                         2D          3D
```

Esto permite ejecutar cada parte del estudio de forma independiente.

---

# 📂 Estructura del proyecto

```text
Teor-a-Electromagn-tica-ll/
│
├── 🧲 faraday_moving/
│   ├── main.py
│   ├── gui.py
│   ├── physics.py
│   ├── solver.py
│   ├── visualization.py
│   └── requirements.txt
│
├── ☁️ gaussian_cloud/
│   ├── main.py
│   ├── gui.py
│   ├── physics.py
│   ├── solver.py
│   ├── visualization.py
│   └── requirements.txt
│
├── 🧭 Gauss's_law_for_magnetism/
│   │
│   ├── analysis/
│   ├── fields/
│   ├── numerical/
│   ├── physics/
│   ├── simulations/
│   ├── sources/
│   ├── tests/
│   ├── visualization/
│   │
│   ├── config.py
│   ├── main.py
│   └── requirements.txt
│
└── README.md
```

---

# ⚙️ Filosofía del proyecto

La idea no es únicamente obtener una solución numérica.

El proyecto busca establecer un flujo completo:

```text
       📐 MODELO FÍSICO
              │
              ▼
       🧮 MODELO MATEMÁTICO
              │
              ▼
       💻 MÉTODO NUMÉRICO
              │
              ▼
       🔬 SIMULACIÓN
              │
              ▼
       📊 ANÁLISIS
              │
              ▼
       🎨 VISUALIZACIÓN
              │
              ▼
       ✅ VERIFICACIÓN
```

De esta manera, la computación se utiliza como una herramienta para
**interpretar y verificar los modelos electromagnéticos**.

---

# 🎯 Objetivos

* Implementar modelos computacionales relacionados con las ecuaciones
  de Maxwell.
* Resolver numéricamente problemas electromagnéticos.
* Comparar resultados numéricos con soluciones analíticas.
* Analizar errores y convergencia.
* Visualizar campos eléctricos y magnéticos.
* Generar representaciones 2D y 3D.
* Utilizar animaciones para facilitar la interpretación física.

---

# 🧰 Herramientas

El proyecto utiliza principalmente herramientas de **computación científica
en Python**, junto con librerías para cálculo numérico, análisis,
visualización y desarrollo de interfaces.

```text
🐍 Python
│
├── 🔢 Computación numérica
├── 📊 Análisis de datos
├── 🎨 Visualización
├── 🖥️ Interfaces gráficas
└── 🎥 Animaciones
```

---

# 👩‍🔬 Autores

<div align="center">

### Laura Arboleda · Sergio Monroy · Valeria Sabogal

**Departamento de Física**
**Universidad del Tolima**
Ibagué, Colombia 🇨🇴

<br>

⚡ 🧲 ☁️ 🧭

</div>

---

# 📚 Asignatura

<div align="center">

**Teoría Electromagnética II**

Proyecto académico de simulación, visualización y análisis computacional
de las **Ecuaciones de Maxwell**.

</div>
