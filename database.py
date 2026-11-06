import os
import sqlite3
from cv2 import imwrite
from datetime import datetime

DB_PATH = "historial_animales.db"
STORAGE_DIR = "storage_media"
CAPTURES_DIR = "storage_capturas"


def inicializar_bd():
    """Crea las carpetas y tablas necesarias en SQLite."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Tabla para archivos multimedia principales
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_archivo TEXT,
            tipo TEXT,
            fecha TEXT,
            ruta_guardada TEXT
        )
    """)
    # Tabla para recortes automáticos de detecciones
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detecciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clase TEXT,
            confianza REAL,
            fecha TEXT,
            ruta_captura TEXT
        )
    """)
    conn.commit()
    conn.close()


def guardar_archivo_y_registro(archivo_subido, tipo_archivo):
    inicializar_bd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_original = archivo_subido.name
    nombre_unico = f"{timestamp}_{nombre_original}"
    ruta_destino = os.path.join(STORAGE_DIR, nombre_unico)

    with open(ruta_destino, "wb") as f:
        f.write(archivo_subido.getbuffer())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO registros (nombre_archivo, tipo, fecha, ruta_guardada)"
        " VALUES (?, ?, ?, ?)",
        (
            nombre_original,
            tipo_archivo,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ruta_destino,
        ),
    )
    conn.commit()
    conn.close()
    return ruta_destino


def guardar_captura(imagen_cv2, clase_nombre, confianza):
    """Guarda el recorte de la mascota detectada en disco y registra en la BD."""
    inicializar_bd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre_archivo = f"{clase_nombre}_{timestamp}.jpg"
    ruta_captura = os.path.join(CAPTURES_DIR, nombre_archivo)
    imwrite(ruta_captura, imagen_cv2)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO detecciones (clase, confianza, fecha, ruta_captura)"
        " VALUES (?, ?, ?, ?)",
        (
            clase_nombre,
            float(confianza),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ruta_captura,
        ),
    )
    conn.commit()
    conn.close()
    return ruta_captura


def obtener_historial():
    inicializar_bd()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, nombre_archivo, tipo, fecha FROM registros ORDER BY id DESC"
    )
    filas = cursor.fetchall()
    conn.close()
    return filas


def obtener_capturas_registradas():
    inicializar_bd()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, clase, confianza, fecha, ruta_captura FROM detecciones ORDER"
        " BY id DESC"
    )
    filas = cursor.fetchall()
    conn.close()
    return filas