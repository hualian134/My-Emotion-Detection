import keras._tf_keras.keras as ks
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Activation, Flatten
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import os

#print(ks.__version__);
num_classes=7
img_rows, img_cols = 48, 48
batch_size = 16 # 8,16,32,64,128

train_data_dir = r'data/train'
validation_data_dir = r'data/test'

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=30, 
                                   shear_range=0.3, 
                                   zoom_range=0.3, 
                                   width_shift_range=0.4, 
                                   height_shift_range=0.4, 
                                   horizontal_flip=True, 
                                   vertical_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    color_mode='grayscale',
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)  

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              color_mode='grayscale',
                                                              target_size=(img_rows, img_cols),
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              shuffle=True)


model = Sequential()

# Block-1
model.add(Conv2D(32, (3, 3), padding='same',kernel_initializer='he_normal',input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same',kernel_initializer='he_normal',input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-2
model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-3   
model.add(Conv2D(128, (3, 3), padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-4
model.add(Conv2D(256, (3, 3), padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())

from keras._tf_keras.keras.optimizers import RMSprop, SGD, Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'Emotion_little_vgg.h5',
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            restore_best_weights=True
                            )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=3,
                                verbose=1,
                                min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

nb_train_samples = 28709
nb_validation_samples = 7178
epochs=25

history=model.fit(train_generator,
                  steps_per_epoch=nb_train_samples // batch_size,
                  epochs=epochs,
                  callbacks=callbacks,
                  validation_data=validation_generator,
                  validation_steps=nb_validation_samples//batch_size)


