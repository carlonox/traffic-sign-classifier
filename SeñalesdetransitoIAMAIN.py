import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator


################# Parametros #####################

path = 'myData'  # archivo con todas las clases
labelFile = 'labels.csv'  # archivo con todos los nombres de las clases
batch_size_val = 50  # cuantos procesar al mismo tiempo
steps_per_epoch_val = 20000
epochs_val = 20 #por cuantas iteraciones pasara
imageDimesions = (32, 32, 3)
testRatio = 0.2  # usa el 20% de las imagenes para el testeo
validationRatio = 0.2  # el restante despues de quitar el 20% se usara para validacion
###################################################


############################### Importo las imagenes
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

############################### Split Datos
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
steps_per_epoch_val = len(X_train)//batch_size_val
validation_steps = len(X_test)//batch_size_val
# X_train = EL ARRAY DE IMAGENES PARA ENTRENAR
# y_train = LAS ID DE LAS CLASES CORRESPONDIENTES



############################### LEER EL ARCHIVO CSV
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

############################### MOSTRAR IMAGENES DE EJEMPLO DE TODAS LAS CLASES
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

############################### MOSTRAR UN GRAFICO DE BARRAS MOSTRANDO EL NUMERO DE ELEMENTOS POR CADA CATEGORIA
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribucion del dataset de entrenamiento")
plt.xlabel("Numero de clases")
plt.ylabel("Numero de imagenes")
plt.show()


############################### PREPROCESAMIENTO DE IMAGENES

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # CONVERTIR A ESCALA DE GRISES
    img = equalize(img)  # STANDARDIZAR LA ILUMINACION EN LA IMAGEN
    img = img / 255  # NORMALLIZAR LOS VALORES ENTRE 0 Y 1 EN VEZ DE 0 Y 255
    return img


X_train = np.array(list(map(preprocessing, X_train)))  # Iterar y preprocesar las imagenes
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("GrayScale Images",
           X_train[random.randint(0, len(X_train) - 1)])  # Revisa si el entrenamiento esta bien hecho

############################### ANADIR UNA PROFUNDIDAD DE 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

############################### AUMENTAR IMAGENES PARA HACERAS GENERICAS
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                             height_shift_range=0.1,
                             zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                             shear_range=0.1,  # MAGNITUD DEL ANGULO DE CORTE
                             rotation_range=10)  # GRADOS
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train,
                       batch_size=20)  # Sicita a el generadpr de datos para dar el tamañio de lotes
X_batch, y_batch = next(batches)

# mostrar ejemplos de imagenes aumentadas
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


############################### Modelo CNN
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)  # Kernel que mueve la imagen alrededor para obtener las caracteristicas
    
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)  #Reduce la escala de el mapa de razgos para generalizar mas y reducir sobreajustes
    no_Of_Nodes = 500  # NUMERO. DE NODOS EN CAPAS OCULTAS
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))  # Añadiendo mas capas convolucionales = Menos rasgos pero incrementa la presicion
    model.add(MaxPooling2D(pool_size=size_of_pool))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))  # Nodos de entrada que se caen con cada actualizacion, 1 todos, 0 ninguno
    model.add(Dense(noOfClasses, activation='softmax'))  # Capa de salida
    # COMPILE MODEL
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


############################### Entrenamiento
model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])


model.save('my_model.h5')  # crea un archivo hdf5 'my_model.h5'
model = load_model('my_model.h5')

cv2.waitKey(0)