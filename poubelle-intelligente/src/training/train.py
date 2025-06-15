from src.data.dataset_loader import DatasetLoader
from src.models.cnn import CNN
from tensorflow.keras.callbacks import ModelCheckpoint

class Trainer:
    def __init__(self, dataset_path, model_save_path, batch_size=32, epochs=10):
        self.dataset_loader = DatasetLoader(dataset_path)
        self.model = CNN()
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_save_path = model_save_path

    def train(self):
        train_data, val_data = self.dataset_loader.load_data()
        checkpoint = ModelCheckpoint(self.model_save_path, save_best_only=True, monitor='val_loss', mode='min')
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[checkpoint]
        )
        return history

if __name__ == "__main__":
    dataset_path = 'data/processed'  # Update with the actual path to processed data
    model_save_path = 'models/model_cnn.h5'
    
    trainer = Trainer(dataset_path, model_save_path)
    trainer.train()
    
    
# def evaluate(self, test_dir, batch_size=32):
#     self.compile()  # <-- Ajouter cette ligne
#     test_datagen = ImageDataGenerator(rescale=1./255)
#     test_generator = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=self.input_shape[:2],
#         batch_size=batch_size,
#         class_mode='binary',
#         shuffle=False
#     )
#     self.model.load_weights(self.model_path)
#     loss, accuracy = self.model.evaluate(test_generator)
#     print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
#     return loss, accuracy

    
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os

# class WasteClassifierModel:
#     def __init__(self, input_shape=(224, 224, 3), model_path='models/model_cnn.h5'):
#         self.input_shape = input_shape
#         self.model_path = model_path
#         self.model = self._build_model()

#     def _build_model(self):
#         model = Sequential()
#         model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(32, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(64, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(128, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(128, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))  # Pour classification binaire
#         return model

#     def compile(self):
#         self.model.compile(
#             optimizer=RMSprop(learning_rate=0.001),
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )

#     def train(self, train_dir, epochs=50, batch_size=32, validation_split=0.2):
#         # Assure-toi que le dossier de sauvegarde existe
#         os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

#         # Création d'un générateur pour l'entraînement avec fraction de validation
#         train_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
#         train_generator = train_datagen.flow_from_directory(
#             train_dir,
#             target_size=self.input_shape[:2],
#             batch_size=batch_size,
#             class_mode='binary',
#             subset='training'
#         )
#         val_generator = train_datagen.flow_from_directory(
#             train_dir,
#             target_size=self.input_shape[:2],
#             batch_size=batch_size,
#             class_mode='binary',
#             subset='validation'
#         )

#         callbacks = [
#             ModelCheckpoint(filepath=self.model_path, save_best_only=True, monitor='val_loss', mode='min'),
#             EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
#         ]

#         self.model.fit(
#             train_generator,
#             epochs=epochs,
#             validation_data=val_generator,
#             callbacks=callbacks
#         )
#         self.model.save(self.model_path)

#     def evaluate(self, test_dir, batch_size=32):
#         test_datagen = ImageDataGenerator(rescale=1./255)
#         test_generator = test_datagen.flow_from_directory(
#             test_dir,
#             target_size=self.input_shape[:2],
#             batch_size=batch_size,
#             class_mode='binary',
#             shuffle=False
#         )
#         self.model.load_weights(self.model_path)
#         loss, accuracy = self.model.evaluate(test_generator)
#         print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
#         return loss, accuracy

# # Exemple d'utilisation
# if __name__ == "__main__":
#     model = WasteClassifierModel()
#     model.compile()
#     model.train('data/DATASET/TRAIN', epochs=50, batch_size=32, validation_split=0.2)
#     model.evaluate('data/DATASET/TEST')