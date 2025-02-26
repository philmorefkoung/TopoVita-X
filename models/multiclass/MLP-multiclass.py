# Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

num_classes = 3 

labels_train = to_categorical(labels_train, num_classes=num_classes)
labels_val = to_categorical(labels_val, num_classes=num_classes)
labels_test = to_categorical(labels_test, num_classes=num_classes)

# Convert into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((betti_train, labels_train))
val_dataset = tf.data.Dataset.from_tensor_slices((betti_val, labels_val))
test_dataset = tf.data.Dataset.from_tensor_slices((betti_test, labels_test))

# Set batch size
batch_size = 128

# Shuffle and batch dataset
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

model = Sequential([
    Input(shape=(400,)),        

    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

def lr_schedule(epoch):
    lr = 0.0001
    if epoch < 50:
        return lr
    elif epoch < 75:
        return lr * 0.1
    else:
        return lr * 0.1 * 0.1  
        
lr_scheduler = LearningRateScheduler(lr_schedule)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')])

# train model
model.fit(train_dataset, 
              epochs=300,
              callbacks=[lr_scheduler, early_stopping], 
              validation_data=val_dataset)

test_loss, test_acc, test_auc, test_precision, test_recall= model.evaluate(test_dataset)

print(f'Test accuracy: {test_acc}')
print(f'Test AUC: {test_auc}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
