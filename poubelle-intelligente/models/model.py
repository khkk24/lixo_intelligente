import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = 'models/full_model.keras'

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Pour binaire
    return model

def train_model():
    os.makedirs('models', exist_ok=True)
    
    model = create_model()
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        'data/DATASET/TRAIN',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'data/DATASET/TRAIN',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    callbacks = [
        ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    ]

    model.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print(f"✅ Modèle sauvegardé dans : {MODEL_PATH}")

def test_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le modèle entraîné est introuvable à {MODEL_PATH}")

    model = load_model(MODEL_PATH)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'data/DATASET/TEST',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    loss, accuracy = model.evaluate(test_generator)
    print(f"🧪 Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()
    test_model()

    
