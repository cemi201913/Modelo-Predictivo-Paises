
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image


# ==============================================================================
# CARGA DEL MODELO Y CLASES
# ==============================================================================

model = tf.keras.models.load_model(
    "modelo_final.keras"
)

with open("class_names.json", "r") as f:
    class_names = json.load(f)


# ==============================================================================
# CONFIGURACIÓN DE STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="GeoClassifier AI",
    page_icon="🌍"
)

st.title(
    "🌍 Clasificador de Países mediante Inteligencia Artificial"
)

st.write(
    "Sube una imagen y el modelo predice el país más probable."
)


# ==============================================================================
# CARGA DE IMAGEN
# ==============================================================================

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Imagen seleccionada",
        use_container_width=True
    )


    img_resized = img.resize(
        (224,224)
    )

    img_array = image.img_to_array(
        img_resized
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0


    prediction = model.predict(
        img_array
    )


    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100


    st.subheader(
        "🌎 Resultado de la predicción"
    )


    st.success(
        f"País predicho: {class_names[predicted_class]}"
    )


    st.write(
        f"Confianza del modelo: {confidence:.2f}%"
    )


    st.subheader(
        "Top 5 predicciones"
    )


    top5_indices = np.argsort(
        prediction[0]
    )[-5:][::-1]


    for idx in top5_indices:

        st.write(
            f"{class_names[idx]}: {prediction[0][idx]*100:.2f}%"
        )
