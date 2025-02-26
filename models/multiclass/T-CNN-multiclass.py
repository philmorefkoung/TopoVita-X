# Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC, Precision, Recall
import pandas as pd 

num_classes = 3 # 3 for NIAID Dataset

labels_train = to_categorical(labels_train, num_classes=num_classes)
labels_val = to_categorical(labels_val, num_classes=num_classes)
labels_test = to_categorical(labels_test, num_classes=num_classes)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(((image_train, betti_train), labels_train))
val_dataset = tf.data.Dataset.from_tensor_slices(((image_val, betti_val), labels_val))
test_dataset = tf.data.Dataset.from_tensor_slices(((image_test, betti_test), labels_test))

# Set batch size
batch_size = 128

# Shuffle and batch dataset
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Load model
base_model = tf.keras.applications.ResNet50( # Swap between DenseNet121 and Xception backbones
    weights= 'imagenet',  # pretrained for full dataset, None for limited data setting
    include_top=False,   
    input_shape=(128, 128, 3) # 128x128
)

# Freeze base model
base_model.trainable = False

# Create model
inputs = tf.keras.Input(shape=(128, 128, 3))

x = base_model(inputs, training=False)  

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.3)(x)

betti_input = tf.keras.Input(shape=(400,))

y = layers.Dense(256)(betti_input)
y = layers.BatchNormalization()(y)
y = layers.ReLU()(y)
y = layers.Dropout(0.3)(y)

y = layers.Dense(128)(y)
y = layers.BatchNormalization()(y)
y = layers.ReLU()(y)
y = layers.Dropout(0.3)(y)

y = layers.Dense(128)(y)
y = layers.BatchNormalization()(y)
y = layers.ReLU()(y)
y = layers.Dropout(0.3)(y)

combined = layers.Concatenate()([x, y])
combined = layers.BatchNormalization()(combined)
combined = layers.ReLU()(combined)
combined = layers.Dropout(0.2)(combined)

combined = layers.Dense(256)(combined)
combined = layers.BatchNormalization()(combined)
combined = layers.ReLU()(combined)
combined = layers.Dropout(0.2)(combined)

combined = layers.Dense(256)(combined)
combined = layers.BatchNormalization()(combined)
combined = layers.ReLU()(combined)
combined = layers.Dropout(0.2)(combined)

output = layers.Dense(num_classes, activation='softmax')(combined)

model = tf.keras.Model(inputs=[inputs, betti_input], outputs=output)

def custom_lr_schedule(epoch):
    if epoch < 50:
        return 0.0001
    elif epoch < 75:
        return 0.0001 * 0.1
    else:
       return 0.0001 * 0.1 * 0.1 

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(custom_lr_schedule)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

# Train model
history = model.fit(
    train_dataset,
    epochs=300,
    callbacks=[lr_scheduler, early_stop],
    validation_data=val_dataset
)

test_loss, test_acc, test_auc, test_precision, test_recall= model.evaluate(test_dataset)

print(f'Test accuracy: {test_acc}')
print(f'Test AUC: {test_auc}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
