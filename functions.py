import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Unpickle function given by the website of the dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# TASK 1 : 

# Load dataset function
def load_cifar10_dataset(data_dir):
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        batch = unpickle(f"{data_dir}/data_batch_{i}")
        x_train.append(batch[b'data'])
        y_train.extend(batch[b'labels'])
    
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    
    test_batch = unpickle(f"{data_dir}/test_batch")
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    #TASK 3 :  1. Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    #TASK 4 : 
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

# TASK 2 
def plot_first_images(x_train, y_train, label_names, num_images):

    #TASK 4 update : it converts the size dynamically
    plt.figure(figsize=(15, int(1.5 * num_images / 5))) 
    
    for i in range(num_images):
        plt.subplot(num_images // 5 + 1, 5, i + 1)
        plt.imshow(x_train[i])

        label_index = np.argmax(y_train[i])
        plt.title(label_names[label_index].decode('utf-8'))
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


#TASK 5 : 

def count_class_occurrences(y_train):
    class_counts = np.sum(y_train, axis=0)
    return class_counts

def plot_class_distribution(class_counts, label_names):
    plt.figure(figsize=(10, 6))
    plt.bar(label_names, class_counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of occurrences')
    plt.title('Distribution of Classes in CIFAR-10 Dataset')
    plt.xticks(rotation=45)
    plt.show()