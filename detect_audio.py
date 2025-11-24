import sounddevice as sd
import numpy as np
import time
import tensorflow as tf
# hi bro
# === Settings ===
MODEL_PATH = "soundclassifier_with_metadata.tflite"
LABELS_PATH = "labels.txt"
SAMPLERATE = 16000
DURATION = 44032 / SAMPLERATE  # ≈ 2.752 seconds

# === Load labels ===
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# === Load model ===
print("📦 Loading model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded successfully!")

# === Prediction function ===
def predict(audio):
    audio = np.array(audio, dtype=np.float32)
    audio = np.expand_dims(audio, axis=0)
    interpreter.set_tensor(input_details[0]['index'], audio)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    top_idx = np.argmax(output)
    return labels[top_idx], output[top_idx]

# === Main Loop ===
print("🎤 Listening for sounds... Press Ctrl+C to stop.")
try:
    while True:
        audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if len(audio) != 44032:
            print("⚠️ Audio length mismatch, skipping...")
            continue

        label, confidence = predict(audio)
        if confidence > 0.75:
            print(f"✅ Detected: {label} ({confidence:.2f})")
        else:
            print("🔎 Low confidence, listening again...")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
