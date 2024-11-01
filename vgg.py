import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib as plt
import matplotlib.pyplot as plt


def create_vgg_like_architecture(num_blocks, input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()
    
    for _ in range(num_blocks):
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), 
                        batch_size=batch_size, verbose=2)
    return history

def evaluate_model(model, x_test, y_test, model_name):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")
    return test_acc

def plot_history(history, model_name):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()