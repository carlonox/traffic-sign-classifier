# 🚦 Traffic Sign Recognition with AI (Real-Time)

> **Real-time traffic sign recognition system** using a webcam and a CNN model trained in Python. Ideal for computer vision, artificial intelligence, or autonomous vehicle projects.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Description

This project uses **Keras/TensorFlow** to train a Convolutional Neural Network (CNN) model that classifies traffic signs. It then leverages **OpenCV** to capture real-time video from your webcam and detect signs in front of the lens.

✅ **Key Features:**
- Train the model with a custom dataset (`myData/`)
- Image preprocessing (grayscale, equalization, normalization)
- Data augmentation to improve generalization
- Real-time detection with configurable confidence threshold
- Visual interface displaying sign name and probability

---

## 🛠️ Technologies Used

| Category         | Technologies                                                              |
|------------------|---------------------------------------------------------------------------|
| **Language**     | Python 3.x                                                                |
| **Libraries**    | OpenCV, Keras, TensorFlow, NumPy, Pandas, Matplotlib, scikit-learn        |
| **Dataset**      | Custom (43 classes, structure similar to GTSRB)                           |
| **IDE**          | Visual Studio (includes `.sln` and `.pyproj` files)                       |
| **Model**        | Custom CNN with convolutional layers, pooling, dropout, and dense layers  |

---

## 🚀 How to Use the Project

### 1. 📥 Clone the repository

```bash
git clone https://github.com/carlonox/traffic-sign-recognition-ai.git
cd traffic-sign-recognition-ai
```

### 2. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

### 3. 🏋️ Train the model (optional)

```bash
python train.py
```

### 4. 🎥 Run real-time detection

```bash
python detect.py
```

---

## 📁 Project Structure

```
traffic-sign-recognition-ai/
├── myData/               # Custom dataset
├── train.py              # Model training script
├── detect.py             # Real-time detection script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── ...
```

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- Inspired by the German Traffic Sign Recognition Benchmark (GTSRB)
- Built with Keras, TensorFlow, and OpenCV

---

⭐ If you found this project useful, give it a star on GitHub!
