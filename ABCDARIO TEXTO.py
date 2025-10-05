# sign_alphabet.py
# Modo de uso:
#  - Recolectar datos (cámara): python sign_alphabet.py collect --label u --samples 100 --seq_len 30
#  - Recolectar datos (video):  python sign_alphabet.py collect --label a --samples 100 --seq_len 30 --video "ruta/video.mp4"
#  - Entrenar:                  python sign_alphabet.py train   --seq_len 30 --epochs 25
#  - Predecir:                  python sign_alphabet.py
#
# Notas:
#  - Tecla 'r' inicia/parar la grabación de una secuencia en modo cámara.
#  - Tecla 'q' sale de cualquier modo con cámara.
#  - En predict, verás la letra más probable y el texto acumulado.

import os
import pyautogui
import pyttsx3
import json
import time
import argparse
import threading
import numpy as np
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ----------------------------
# Config comunes
# ----------------------------
LABELS = list("abcdefghijklmnñopqrstuvwxyz")  # abecedario completo
DATA_DIR = Path("C:\\Users\morena\Desktop\ABCDARIO\data")
MODEL_DIR = Path("C:\\Users\morena\Desktop\ABCDARIO\data")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "cnn_alphabet.h5"
LABELS_PATH = MODEL_DIR / "labels.json"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

engine = pyttsx3.init()

# Configurar velocidad y volumen (opcional)
engine.setProperty('rate', 150)    # velocidad de lectura (palabras por minuto)
engine.setProperty('volume', 1.0)
def ensure_dirs():
    for lbl in LABELS:
        (DATA_DIR / lbl).mkdir(parents=True, exist_ok=True)


# ----------------------------
# Utilidades de landmarks
# ----------------------------
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
    origin = pts[0].copy()
    pts -= origin
    scale = np.linalg.norm(pts, axis=1).mean() + 1e-6
    pts /= scale
    return pts.reshape(-1)


def draw_hand(image_bgr, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )


# ----------------------------
# Recolección de datos (cámara o video)
# ----------------------------
def collect_mode(args):
    assert args.label in LABELS, f"Etiqueta inválida. Usa: {LABELS}"
    ensure_dirs()

    seq_len = args.seq_len
    samples_needed = args.samples
    out_dir = DATA_DIR / args.label

    source = 0 if args.video is None else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    print(f"[COLLECT] Etiqueta '{args.label}' | muestras a grabar: {samples_needed} | seq_len: {seq_len}")
    if args.video is None:
        print("Modo: CÁMARA en vivo")
        print("Instrucciones:")
        print(" - Coloca la mano dentro del recuadro. Presiona 'r' para grabar una secuencia.")
        print(" - Mantén la seña estable hasta que la barra llegue al final.")
        print(" - 'q' para salir.")
    else:
        print(f"Modo: VIDEO ({args.video})")
        print(" - Procesando automáticamente frames del video...")

    recorder = deque(maxlen=seq_len)
    is_recording = False
    saved = 0
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.video is None:  # cámara
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            draw_hand(frame, results)

            lm = extract_landmarks(results)
            if lm is not None:
                recorder.append(lm)

            # --- cámara en vivo ---
            if args.video is None:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (int(0.1*w), int(0.1*h)), (int(0.9*w), int(0.9*h)), (255, 255, 0), 1)

                bar_len = int((len(recorder) / seq_len) * 300)
                cv2.putText(frame, f"Label: {args.label.upper()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.rectangle(frame, (10, 50), (310, 70), (200, 200, 200), 2)
                cv2.rectangle(frame, (10, 50), (10+bar_len, 70), (0, 200, 0), -1)
                if is_recording:
                    cv2.putText(frame, "Grabando...", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,150,0), 2)

                cv2.imshow("Recolectar - letra "+args.label.upper(), frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    is_recording = not is_recording
                    if not is_recording:
                        recorder.clear()

                if is_recording and len(recorder) == seq_len:
                    arr = np.stack(list(recorder), axis=0).astype(np.float32)
                    fname = out_dir / f"seq_{saved:03d}.npy"
                    np.save(fname, arr)
                    saved += 1
                    recorder.clear()
                    is_recording = False
                    print(f"Guardado {fname}")
                    time.sleep(0.3)
                    if saved >= samples_needed:
                        print("¡Listo! Muestras completadas.")
                        break

            # --- modo video (automático) ---
            else:
                if len(recorder) == seq_len:
                    arr = np.stack(list(recorder), axis=0).astype(np.float32)
                    fname = out_dir / f"seq_{saved:03d}.npy"
                    np.save(fname, arr)
                    saved += 1
                    recorder.clear()
                    print(f"Guardado {fname}")
                    if saved >= samples_needed:
                        print("¡Listo! Muestras completadas desde video.")
                        break

    finally:
        cap.release()
        if args.video is None:
            cv2.destroyAllWindows()
        hands.close()


# ----------------------------
# Dataset
# ----------------------------
def load_dataset(seq_len):
    X, y = [], []
    label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
    for lbl in LABELS:
        folder = DATA_DIR / lbl
        if not folder.exists():
            continue
        for f in sorted(folder.glob("seq_*.npy")):
            arr = np.load(f)
            if arr.shape[0] != seq_len:
                idxs = np.linspace(0, arr.shape[0]-1, seq_len).astype(int)
                arr = arr[idxs]
            X.append(arr)
            y.append(label_to_idx[lbl])
    if not X:
        raise RuntimeError("No hay datos. Recolecta con el modo 'collect'.")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


# ----------------------------
# Modelo
# ----------------------------
def build_model(seq_len, n_features=63, n_classes=len(LABELS)):
    model = Sequential([
        Conv1D(128, kernel_size=5, padding='same', input_shape=(seq_len, n_features)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),

        Conv1D(128, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),

        Conv1D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_mode(args):
    ensure_dirs()
    X, y = load_dataset(args.seq_len)
    num_classes = len(LABELS)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    y_train_oh = to_categorical(y_train, num_classes)
    y_val_oh = to_categorical(y_val, num_classes)

    model = build_model(args.seq_len, n_features=X.shape[2], n_classes=num_classes)
    model.summary(print_fn=lambda x: print("[MODEL] " + x))

    callbacks = [
        ModelCheckpoint(str(MODEL_PATH), monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    test_oh = to_categorical(y_test, num_classes)
    loss, acc = model.evaluate(X_test, test_oh, verbose=0)
    print(f"[TEST] accuracy={acc:.3f} loss={loss:.3f}")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Etiquetas guardadas en: {LABELS_PATH}")


# ----------------------------
# Predicción
# ----------------------------
def predict_mode(args):
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Falta el modelo o el archivo de etiquetas. Entrena primero con 'train'.")

    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    seq_len = args.seq_len
    buffer = deque(maxlen=seq_len)
    sentence = ""  # texto acumulado

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            draw_hand(frame, results)
            lm = extract_landmarks(results)
            if lm is not None:
                buffer.append(lm)

            pred_text = "..."
            if len(buffer) == seq_len:
                x = np.expand_dims(np.stack(list(buffer), axis=0), axis=0)
                probs = model.predict(x, verbose=0)[0]
                idx = int(np.argmax(probs))
                confidence = probs[idx]
                pred_letter = labels[idx].upper()
                pred_text = f"{pred_letter}  ({confidence*100:.1f}%)"

                # agregar al texto si la confianza es alta
                import pyautogui  # asegurate de importar al inicio del archivo

                if confidence > 0.2c:
                    if len(sentence) == 0 or sentence[-1] != pred_letter:
                        sentence += pred_letter
                        texto = sentence

                        # Reproducir audio
                        engine.say(pred_letter)
                        engine.runAndWait()

                        # Escribir la letra reconocida en la ventana activa (teclado en vivo)
                        pyautogui.write(pred_letter)

            # mostrar letra actual
            cv2.putText(frame, "Letra:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, pred_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

            # mostrar texto acumulado
            cv2.putText(frame, "Texto:", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, sentence, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)

            # controles extra
            cv2.putText(frame, "q: salir | c: limpiar", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.imshow("Prediccion - abecedario", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = ""  # limpiar texto

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Deteccion de abecedario con MediaPipe+CNN+OpenCV")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_collect = sub.add_parser("collect", help="Recolectar secuencias para una etiqueta (cámara o video)")
    p_collect.add_argument("--label", type=str, required=True, choices=LABELS)
    p_collect.add_argument("--samples", type=int, default=100)
    p_collect.add_argument("--seq_len", type=int, default=30)
    p_collect.add_argument("--video", type=str, default=None, help= "C:\\Users\morena\Desktop\\videos_para_datos.mp4\LETRA Z.mp4")

    p_train = sub.add_parser("train", help="Entrenar la CNN 1D")
    p_train.add_argument("--seq_len", type=int, default=30)
    p_train.add_argument("--epochs", type=int, default=25)
    p_train.add_argument("--batch_size", type=int, default=32)

    p_predict = sub.add_parser("predict", help="Prediccion en tiempo real")
    p_predict.add_argument("--seq_len", type=int, default=30)

    args = parser.parse_args()

    if args.mode == "collect":
        collect_mode(args)
    elif args.mode == "train":
        train_mode(args)
    elif args.mode == "predict":
        predict_mode(args)


if __name__ == "__main__":
    main()