

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ====== Settings ======
  # or "/Users/.../Desktop/dataset_split"
TRAIN_DIR = os.path.join("/content/dataset_split/dataset_split/train")
VAL_DIR   = os.path.join("/content/dataset_split/dataset_split/val")
TEST_DIR  = os.path.join("/content/dataset_split/dataset_split/test")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5

INITIAL_EPOCHS = 6
FINE_TUNE_EPOCHS = 10
BASE_LR = 1e-3
FINETUNE_LR = 1e-4
# ======================

# ---- Data generators with augmentation (use preprocess_input for MobileNetV2) ----
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# ---- Optional: compute class weights to handle imbalance ----
labels = []
for cls, idx in train_gen.class_indices.items():
    pass
# Build label array for class_weight calculation
y_train = []
for i in range(len(train_gen)):
    _, yb = train_gen[i]
    y_train.extend(np.argmax(yb, axis=1))
y_train = np.array(y_train)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weights)

# ---- Build model: MobileNetV2 backbone (frozen first) ----
base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                         include_top=False, weights='imagenet')
base_model.trainable = False  # freeze initially

inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4))(x)
x = la5yers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(learning_rate=BASE_LR),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---- Callbacks ----
checkpoint_cb = callbacks.ModelCheckpoint("mobilenet_best.h5", save_best_only=True, monitor='val_accuracy', mode='max')
earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# ---- Train top layers first ----
history1 = model.fit(
    train_gen,
    epochs=INITIAL_EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

# ---- Unfreeze some of the base model for fine-tuning ----
base_model.trainable = True
# Optionally unfreeze only the last N layers:
fine_tune_at = len(base_model.layers) - 40  # unfreeze last 40 layers
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= fine_tune_at)

model.compile(optimizer=optimizers.Adam(learning_rate=FINETUNE_LR),
              loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(
    train_gen,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history1.epoch[-1] + 1 if history1.epoch else 0,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

# ---- Evaluate on test set ----
model.load_weights("mobilenet_best.h5")
loss, acc = model.evaluate(test_gen)
print(f"Test accuracy: {acc*100:.2f}%")

# ---- Confusion matrix & classification report ----
# get predictions
pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# ---- Save final model (SavedModel) ----
model.save("mobilenet_drowsiness.keras")

# ---- Convert to TFLite (float16 quantization recommended for mobile speed) ----
def convert_to_tflite(saved_model_dir, tflite_path, quantize=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model to:", tflite_path)

convert_to_tflite("mobilenet_drowsiness_savedmodel", "mobilenet_drowsiness_fp16.tflite", quantize=True)
