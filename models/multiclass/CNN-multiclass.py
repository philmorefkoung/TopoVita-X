# Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

num_classes = 3 # 3 for NIAID

labels_train = to_categorical(labels_train, num_classes=num_classes) 
labels_val = to_categorical(labels_val, num_classes=num_classes)
labels_test = to_categorical(labels_test, num_classes=num_classes)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((image_train, labels_train))
val_dataset = tf.data.Dataset.from_tensor_slices((image_val, labels_val))
test_dataset = tf.data.Dataset.from_tensor_slices((image_test, labels_test))

# Set batch size
batch_size = 128

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Load the model
base_model = tf.keras.applications.ResNet50( # Swap between DenseNet121 and Xception
    weights= 'imagenet',  # Pre-trained for full datasets, 'None' for limited data settings
    include_top=False, 
    input_shape=(128, 128, 3) # 128x128 for all datasets, ALLIDB2 is not a multiclass task
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

outputs = layers.Dense(num_classes, activation='softmax')(x) 

model = tf.keras.Model(inputs, outputs)

def custom_lr_schedule(epoch):
    if epoch < 50:
      return 0.0001
    elif epoch < 75:
      return 0.0001 * 0.1
    else:
      return 0.0001 * 0.1 * 0.1

# LearningRateScheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(custom_lr_schedule)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy','AUC', 'Precision', 'Recall']
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience=5, 
                                              mode='min', 
                                              restore_best_weights=True)

# Train model
history = model.fit(
    train_dataset,
    epochs=300,
    callbacks=[lr_scheduler, early_stop],
    validation_data=val_dataset,
)
result = model.evaluate(test_dataset)

test_loss, test_acc, test_auc, test_precision, test_recall = result

print(f'Test Accuracy: {test_acc}')
print(f'Test AUC: {test_auc}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
