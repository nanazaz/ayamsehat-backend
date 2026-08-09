"""
=============================================================
  ayamsehat.online — Model Training Script (Updated)
  Dataset  : Poultry Diseases Detection (Kaggle)
             struktur: dataset/ClassName/foto.jpg
             (tanpa folder train/val/test — auto-split!)
  Model    : MobileNetV2 (Transfer Learning)
=============================================================

CARA PAKAI:
  Struktur dataset yang dibutuhkan:
    dataset/
      Coccidiosis/    ← foto-foto langsung di sini
      Healthy/
      Newcastle Disease/
      Salmonella/

  Jalankan:
    python train_model.py
"""

import os, json, warnings, shutil, random
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────
CFG = {
    "dataset_dir"    : "./dataset",
    "split_dir"      : "./dataset_split",   # folder sementara hasil split
    "model_save_dir" : "./saved_model",
    "img_size"       : (224, 224),
    "batch_size"     : 32,
    "epochs_warmup"  : 10,
    "epochs_finetune": 20,
    "learning_rate"  : 1e-4,
    "dropout"        : 0.3,
    "seed"           : 42,
    "train_ratio"    : 0.7,
    "val_ratio"      : 0.15,
    # test_ratio      = sisa 0.15
}

CLASS_NAMES = ["Coccidiosis", "Healthy", "Newcastle Disease", "Salmonella"]
NUM_CLASSES  = len(CLASS_NAMES)

os.makedirs(CFG["model_save_dir"], exist_ok=True)
tf.random.set_seed(CFG["seed"])
random.seed(CFG["seed"])

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")


# ─────────────────────────────────────────
#  1. AUTO-SPLIT DATASET
# ─────────────────────────────────────────
def auto_split():
    """
    Baca foto dari dataset/ClassName/
    lalu split ke dataset_split/train|val|test/ClassName/
    """
    split_dir = CFG["split_dir"]

    # Kalau sudah pernah di-split, skip
    if os.path.exists(split_dir):
        print(f"✅ Folder split sudah ada ({split_dir}), skip proses split.")
        return

    print("\n📂 Auto-split dataset → train / val / test ...")
    for cls in CLASS_NAMES:
        src = os.path.join(CFG["dataset_dir"], cls)
        if not os.path.exists(src):
            print(f"⚠️  Folder tidak ditemukan: {src}")
            print(f"    Pastikan nama folder PERSIS: {CLASS_NAMES}")
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {src}")

        # Ambil semua file gambar
        files = [
            f for f in os.listdir(src)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        random.shuffle(files)

        n       = len(files)
        n_train = int(n * CFG["train_ratio"])
        n_val   = int(n * CFG["val_ratio"])

        splits = {
            "train": files[:n_train],
            "val"  : files[n_train:n_train + n_val],
            "test" : files[n_train + n_val:],
        }

        print(f"  {cls:20s}: total={n}, train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

        for split_name, split_files in splits.items():
            dst_dir = os.path.join(split_dir, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in split_files:
                shutil.copy2(os.path.join(src, fname),
                             os.path.join(dst_dir, fname))

    print("✅ Split selesai!\n")


# ─────────────────────────────────────────
#  2. DATA PIPELINE
# ─────────────────────────────────────────
def build_generators():
    train_gen = ImageDataGenerator(
        rescale           = 1.0 / 255,
        rotation_range    = 20,
        width_shift_range = 0.15,
        height_shift_range= 0.15,
        zoom_range        = 0.2,
        horizontal_flip   = True,
        brightness_range  = [0.8, 1.2],
        fill_mode         = "nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    split_dir = CFG["split_dir"]

    train_ds = train_gen.flow_from_directory(
        os.path.join(split_dir, "train"),
        target_size = CFG["img_size"],
        batch_size  = CFG["batch_size"],
        class_mode  = "categorical",
        classes     = CLASS_NAMES,
        seed        = CFG["seed"],
    )
    val_ds = val_gen.flow_from_directory(
        os.path.join(split_dir, "val"),
        target_size = CFG["img_size"],
        batch_size  = CFG["batch_size"],
        class_mode  = "categorical",
        classes     = CLASS_NAMES,
        shuffle     = False,
    )
    test_ds = val_gen.flow_from_directory(
        os.path.join(split_dir, "test"),
        target_size = CFG["img_size"],
        batch_size  = CFG["batch_size"],
        class_mode  = "categorical",
        classes     = CLASS_NAMES,
        shuffle     = False,
    )
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────
#  3. MODEL
# ─────────────────────────────────────────
def build_model():
    base = MobileNetV2(
        input_shape = (*CFG["img_size"], 3),
        include_top = False,
        weights     = "imagenet",
    )
    base.trainable = False

    inputs  = layers.Input(shape=(*CFG["img_size"], 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(CFG["dropout"])(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(CFG["dropout"] / 2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="AyamSehat_MobileNetV2"), base


# ─────────────────────────────────────────
#  4. TRAINING
# ─────────────────────────────────────────
def train():
    print("\n" + "=" * 55)
    print("  ayamsehat.online — Model Training")
    print("=" * 55)

    # Split dulu kalau belum
    auto_split()

    # Load data
    train_ds, val_ds, test_ds = build_generators()
    print(f"Total train : {train_ds.samples} gambar")
    print(f"Total val   : {val_ds.samples} gambar")
    print(f"Total test  : {test_ds.samples} gambar\n")

    # Build model
    model, base = build_model()
    model.summary()

    # Callbacks
    cb = [
        callbacks.ModelCheckpoint(
            filepath       = os.path.join(CFG["model_save_dir"], "best_model.h5"),
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 1,
        ),
        callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 8,
            restore_best_weights = True,
            verbose              = 1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 4,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
        callbacks.CSVLogger(
            os.path.join(CFG["model_save_dir"], "training_log.csv")
        ),
    ]

    # FASE 1: Warmup
    print("\n[FASE 1] Warmup — backbone frozen")
    model.compile(
        optimizer = optimizers.Adam(CFG["learning_rate"]),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    hist1 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = CFG["epochs_warmup"],
        callbacks       = cb,
        verbose         = 1,
    )

    # FASE 2: Fine-tune
    print("\n[FASE 2] Fine-tuning — unfreeze 30 layer terakhir")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer = optimizers.Adam(CFG["learning_rate"] / 10),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    hist2 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = CFG["epochs_finetune"],
        callbacks       = cb,
        verbose         = 1,
    )

    # Evaluasi
    print("\n[EVALUASI] Test set")
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(test_ds), axis=1)
    y_true = test_ds.classes
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix — ayamsehat.online")
    plt.ylabel("Label Asli"); plt.xlabel("Label Prediksi")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["model_save_dir"], "confusion_matrix.png"), dpi=150)

    # Export TFLite
    print("\n[EXPORT] Konversi ke TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(os.path.join(CFG["model_save_dir"], "ayamsehat_model.tflite"), "wb") as f:
        f.write(tflite_model)

    # Label map
    label_map = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    with open(os.path.join(CFG["model_save_dir"], "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    # Plot history
    plot_history(hist1, hist2)
    print("\n✅ Training selesai! Model siap digunakan.")


def plot_history(h1, h2):
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    val   = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]
    ep    = range(1, len(acc) + 1)
    warmup_end = len(h1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ayamsehat.online — Training History", fontsize=14, fontweight="bold")
    for ax, (m, v), title in zip(
        axes, [(acc, val), (loss, vloss)], ["Accuracy", "Loss"]
    ):
        ax.plot(ep, m, label=f"Train {title}", color="#E8A800")
        ax.plot(ep, v, label=f"Val {title}",   color="#E8412A")
        ax.axvline(warmup_end, color="gray", linestyle="--",
                   alpha=0.6, label="Fine-tune start")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["model_save_dir"], "training_history.png"), dpi=150)
    print("✅ Grafik training disimpan.")


if __name__ == "__main__":
    train()
