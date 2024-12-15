# Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import pandas as pd
import numpy as np
                       
# Convert into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((betti_train, labels_train))
val_dataset = tf.data.Dataset.from_tensor_slices((betti_val, labels_val))
test_dataset = tf.data.Dataset.from_tensor_slices((betti_test, labels_test))

# Set batch size
batch_size = 128

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

model = Sequential([
    Input(shape=(400,)), # betti vector for each image should be 400-dimensions 50 for betti0 50 for betti1 for Gray, R, G, B channels
   
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
    
    Dense(1, activation='sigmoid')
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
              loss='binary_crossentropy',
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