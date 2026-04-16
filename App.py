import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Función probabilidades
def getProbabilities(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    return pred[0]

# ================= UI =================
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')

st.title('Reconocimiento de Dígitos escritos a mano')

# Imagen de titulo
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
    caption="Ejemplos de dígitos escritos a mano (MNIST)",
    width=600
)

st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Opciones")

stroke_width = st.sidebar.slider('Ancho de línea', 1, 30, 15)
stroke_color = st.sidebar.color_picker('Color del lápiz', '#FFFFFF')
bg_color = st.sidebar.color_picker('Color del fondo', '#000000')

drawing_mode = st.sidebar.selectbox(
    "Modo de dibujo",
    ("freedraw", "line", "rect", "circle")
)

st.sidebar.markdown("---")
st.sidebar.title("Acerca de:")
st.sidebar.text("Fork de Karen Hernández para InterMultim")
st.sidebar.text("En esta aplicación se evalúa")
st.sidebar.text("la capacidad de una RNA de reconocer") 
st.sidebar.text("dígitos escritos a mano.")
st.sidebar.text("Basado en desarrollo de Vinay Uniyal")

# ================= CANVAS =================
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=400,
    width=400,
    drawing_mode=drawing_mode,
    key="canvas",
)

# ================= BOTÓN =================
if st.button('Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")

        res = predictDigit(img)
        st.header('El Dígito es: ' + str(res))

        probs = getProbabilities(img)

        df = pd.DataFrame({
            'Dígito': list(range(10)),
            'Probabilidad': probs
        })

        st.markdown("### 📊 Probabilidad por cada dígito")
        st.bar_chart(df.set_index('Dígito'))

    else:
        st.header('Por favor dibuja en el canvas el dígito.')
