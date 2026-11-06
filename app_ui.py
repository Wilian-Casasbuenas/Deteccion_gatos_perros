import tempfile
import cv2
from PIL import Image
import streamlit as st

from ai_model import DetectorAnimalesAI
from database import (
    guardar_archivo_y_registro,
    obtener_capturas_registradas,
    obtener_historial,
)

st.set_page_config(
    page_title="Sistema Modular de Mascotas", page_icon="🐾", layout="wide"
)


@st.cache_resource
def cargar_ia():
    return DetectorAnimalesAI("yolov8n.pt")


ia = cargar_ia()

st.title("🐾 Detector Inteligente de Perros y Gatos con Conteo y Capturas")
st.write(
    "Sistema modular avanzado: Conteo independiente por especie y registro"
    " fotográfico automático."
)

# Menú lateral
menu = st.sidebar.selectbox(
    "Menú Principal",
    ("Analizar Archivo", "Historial Multimedia", "Galería de Capturas"),
)
st.sidebar.markdown("---")
filtrar_mascotas = st.sidebar.checkbox("Filtrar solo Perros y Gatos", value=True)
guardar_crops_auto = st.sidebar.checkbox(
    "📸 Guardar recortes automáticos al detectar",
    value=True,
    help=(
        "Guarda una foto recortada de cada perro o gato detectado en la base de"
        " datos."
    ),
)
confianza = st.sidebar.slider("Nivel de Confianza Mínimo", 0.1, 1.0, 0.5, 0.05)

if menu == "Analizar Archivo":
    tipo_modo = st.radio("Selecciona el recurso:", ("Imagen", "Video"))

    if tipo_modo == "Imagen":
        archivo = st.file_uploader(
            "Sube una imagen...", type=["jpg", "jpeg", "png"]
        )

        if archivo is not None:
            guardar_archivo_y_registro(archivo, "Imagen")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagen Original")
                imagen_pil = Image.open(archivo)
                st.image(imagen_pil, use_container_width=True)

            with col2:
                st.subheader("Resultado y Conteo")
                if st.button("🚀 Ejecutar Reconocimiento"):
                    with st.spinner("Analizando..."):
                        imagen_resultado, conteo, crops = ia.procesar_imagen(
                            imagen_pil,
                            confianza=confianza,
                            solo_perros_gatos=filtrar_mascotas,
                            guardar_crops=guardar_crops_auto,
                        )
                        st.image(imagen_resultado, use_container_width=True)

                        # Métricas en pantalla
                        m1, m2 = st.columns(2)
                        m1.metric("🐶 Perros Detectados", conteo["perros"])
                        m2.metric("🐱 Gatos Detectados", conteo["gatos"])

                        if crops:
                            st.success(
                                f"¡Se guardaron {len(crops)} recortes automáticamente en la"
                                " base de datos!"
                            )
                        else:
                            st.info("No se realizaron nuevas capturas automáticas.")

    elif tipo_modo == "Video":
        archivo = st.file_uploader("Sube un video...", type=["mp4", "avi", "mov"])

        if archivo is not None:
            guardar_archivo_y_registro(archivo, "Video")

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(archivo.read())
            st.video(tfile.name)

            if st.button("▶️ Procesar Video"):
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                metricas_placeholder = st.empty()
                barra = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                actual = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_procesado, conteo, _ = ia.procesar_frame(
                        frame,
                        confianza=confianza,
                        solo_perros_gatos=filtrar_mascotas,
                        guardar_crops=guardar_crops_auto,
                    )
                    stframe.image(frame_procesado, channels="RGB", use_container_width=True)

                    with metricas_placeholder.container():
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("🐶 Perros en Frame", conteo["perros"])
                        col_m2.metric("🐱 Gatos en Frame", conteo["gatos"])

                    actual += 1
                    if total_frames > 0:
                        barra.progress(min(actual / total_frames, 1.0))

                cap.release()
                st.success("¡Video procesado con éxito!")

elif menu == "Historial Multimedia":
    st.subheader("📁 Archivos Originales Subidos")
    historial = obtener_historial()
    if historial:
        for item in historial:
            st.markdown(
                f"- **ID:** {item[0]} | **Archivo:** `{item[1]}` | **Tipo:**"
                f" `{item[2]}` | **Fecha:** `{item[3]}`"
            )
    else:
        st.info("No hay archivos multimedia registrados.")

elif menu == "Galería de Capturas":
    st.subheader("📸 Recortes y Detecciones Guardadas Automáticamente")
    capturas = obtener_capturas_registradas()
    if capturas:
        cols = st.columns(3)
        for i, cap in enumerate(capturas):
            # cap: (id, clase, confianza, fecha, ruta_captura)
            with cols[i % 3]:
                if os.path.exists(cap[4]):
                    st.image(
                        cap[4],
                        caption=(
                            f"Clase: {cap[1].upper()} | Conf: {cap[2]:.2f}\nFecha:"
                            f" {cap[3]}"
                        ),
                        use_container_width=True,
                    )
    else:
        st.info(
            "Aún no hay recortes guardados. Activa la opción de capturas y realiza"
            " un análisis."
        )