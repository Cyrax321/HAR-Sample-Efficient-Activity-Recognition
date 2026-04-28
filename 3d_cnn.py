#  Install & Import Dependencies
# !pip install kagglehub reportlab scikit-learn opencv-python-headless tqdm seaborn --quiet

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import logging
from datetime import datetime
from google.colab import drive
import os
import zipfile
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

#  Setup Google Drive
drive.mount('/content/drive', force_remount=True)
project_root = '/content/drive/MyDrive/3D_CNN_HAR_Project'
os.makedirs(project_root, exist_ok=True)

paths = {
    'log_file': os.path.join(project_root, 'training_log.txt'),
    'model': os.path.join(project_root, 'best_3dcnn_model.keras'),
    'history': os.path.join(project_root, 'training_history.json'),
    'plot': os.path.join(project_root, 'training_plots.png'),
    'cm': os.path.join(project_root, 'confusion_matrix.png'),
    'f1': os.path.join(project_root, 'f1_scores.png'),
    'report': os.path.join(project_root, 'classification_report.json'),
    'pdf': os.path.join(project_root, 'Model_Report.pdf')
}

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(paths['log_file']), logging.StreamHandler()])

#  Download + Load Dataset
logging.info(" Downloading dataset...")
dataset_dir = kagglehub.dataset_download("sharjeelmazhar/human-activity-recognition-video-dataset")
extract_path = next((root for root, dirs, _ in os.walk(dataset_dir) if len(dirs) > 1), None)
if extract_path is None:
    raise FileNotFoundError("Dataset root not found.")

classes_to_use = sorted([d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))])
class_to_label = {cls: idx for idx, cls in enumerate(classes_to_use)}
label_to_class = {v: k for k, v in class_to_label.items()}

#  Preprocess Videos
FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH = 30, 64, 64

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, FRAME_COUNT).astype(int)
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    while len(frames) < FRAME_COUNT:
        frames.append(frames[-1] if frames else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))
    return np.array(frames)

X, y = [], []
LIMIT_PER_CLASS = 50
for cls in classes_to_use:
    class_path = os.path.join(extract_path, cls)
    video_files = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi'))][:LIMIT_PER_CLASS]
    for file in tqdm(video_files, desc=f"Loading {cls}"):
        try:
            frames = load_video_frames(os.path.join(class_path, file))
            X.append(frames)
            y.append(class_to_label[cls])
        except Exception as e:
            logging.warning(f" Failed: {file} => {e}")

X = np.array(X).astype('float32') / 255.0
y_cat = tf.keras.utils.to_categorical(y, num_classes=len(classes_to_use))
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)

#  Build Model
def build_model():
    model = Sequential([
        Conv3D(32, (3,3,3), activation='relu', padding='same', input_shape=(FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 3)),
        BatchNormalization(), MaxPooling3D((1,2,2)),
        Conv3D(64, (3,3,3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling3D((2,2,2)),
        Conv3D(128, (3,3,3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling3D((2,2,2)),
        Flatten(), Dense(256, activation='relu'), Dropout(0.5),
        Dense(len(classes_to_use), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

#  Training Loop with Accuracy Tracking
EPOCHS = 30
BATCH_SIZE = 8
best_val_acc = 0

train_accs, val_accs, test_accs = [], [], []

for epoch in range(1, EPOCHS + 1):
    print(f"\n Epoch {epoch}/{EPOCHS}")
    
    # Train for one epoch
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=1, batch_size=BATCH_SIZE, verbose=1)
    
    train_acc = history.history['accuracy'][0]
    val_acc = history.history['val_accuracy'][0]
    
    y_test_pred_probs = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    test_acc = np.mean(y_test_pred == y_test_true)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    print(f" Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save(paths['model'])
        print(" Best model saved.")

#  Save History
history_data = {
    "train_accuracy": train_accs,
    "val_accuracy": val_accs,
    "test_accuracy": test_accs
}
with open(paths['history'], 'w') as f:
    json.dump(history_data, f)

#  Classification Report & Confusion Matrix
print("\n Classification Report:")
report = classification_report(y_test_true, y_test_pred, target_names=classes_to_use, output_dict=True)
print(classification_report(y_test_true, y_test_pred, target_names=classes_to_use))

with open(paths['report'], 'w') as f:
    json.dump(report, f, indent=4)

# Confusion Matrix
cm = confusion_matrix(y_test_true, y_test_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes_to_use, yticklabels=classes_to_use, cmap='Blues')
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig(paths['cm']); plt.close()

#  Accuracy Plot (Train, Val, Test)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.plot(test_accs, label='Test Acc')
plt.legend(); plt.title("Accuracy")

# Placeholder for loss (if needed)
plt.subplot(1,2,2)
plt.plot([0]*EPOCHS, label='Train Loss')  # Placeholder
plt.plot([0]*EPOCHS, label='Val Loss')    # Placeholder
plt.legend(); plt.title("Loss (placeholder)")
plt.savefig(paths['plot']); plt.close()

#  F1 Score per Class
f1_scores = [report[cls]['f1-score'] for cls in classes_to_use]
plt.figure(figsize=(8,6))
sns.barplot(x=classes_to_use, y=f1_scores)
plt.title("F1 Scores per Class")
plt.ylabel("F1-score")
plt.xticks(rotation=45)
plt.savefig(paths['f1']); plt.close()
