# 🚦 Traffic Sign Recognition with AI (Real-Time)

> **Sistema de reconocimiento de señales de tránsito en tiempo real** usando una cámara web y un modelo CNN entrenado en Python. Ideal para proyectos de visión por computadora, inteligencia artificial o vehículos autónomos.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)


---

## 📌 Descripción

Este proyecto utiliza **Keras/TensorFlow** para entrenar un modelo de red neuronal convolucional (CNN) que clasifica señales de tránsito. Luego, usa **OpenCV** para capturar video en tiempo real desde tu cámara web y detectar señales frente a la lente.

✅ **Características principales:**
- Entrenamiento del modelo con dataset personalizado (`myData/`)
- Preprocesamiento de imágenes (escala de grises, equalización, normalización)
- Aumento de datos para mejorar la generalización
- Detección en tiempo real con umbral de confianza configurable
- Interfaz visual con nombre de la señal y probabilidad

---

## 🛠️ Tecnologías Utilizadas

| Categoría       | Tecnologías                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Lenguaje**   | Python 3.x                                                                  |
| **Librerías**  | OpenCV, Keras, TensorFlow, NumPy, Pandas, Matplotlib, scikit-learn          |
| **Dataset**    | Personalizado (43 clases, estructura similar a GTSRB)                        |
| **IDE**        | Visual Studio (archivos `.sln` y `.pyproj` incluidos)                       |
| **Modelo**     | CNN personalizada con capas convolucionales, pooling, dropout y dense       |

---

## 🚀 Cómo Usar el Proyecto

### 1. 📥 Clonar el repositorio

```bash
git clone https://github.com/carlonox/traffic-sign-recognition-ai.git
cd traffic-sign-recognition-ai
