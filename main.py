import functions as fct
import vgg as dnn
import lenet5
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop, SGD, Adam 

data_dir = "data/cifar-10-batches-py"
(x_train, y_train), (x_test, y_test) = fct.load_cifar10_dataset(data_dir)

def display_menu(): 
    print("\nMenu:")
    print("1. Display the task 1 results.")
    print("2. Display the task 2 results.")
    print("3. Display the task 5 results.")
    print("4. Start the VGG model.")
    print("5. Start the LeNet5 model.")
    print("0. Exit")
     
def main(): 
    while True:
        display_menu()
        try:
            menu_nb = int(input("Enter your choice: "))  
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue 

        if menu_nb == 1: 
            print("Displaying the shape of the training and test sets to understand the dimensions of the data.")
            print("Training set shape (images):", x_train.shape)
            print("Training set shape (labels):", y_train.shape)
            print("Test set shape (images):", x_test.shape)
            print("Test set shape (labels):", y_test.shape)

        elif menu_nb == 2: 
            num_image = int(input("how many images do you want to see ?"))
            print("Displaying the first",num_image, "images of the training set.")
            meta = fct.unpickle(f"{data_dir}/batches.meta")
            label_names = meta[b'label_names']
            fct.plot_first_images(x_train, y_train, label_names,num_image)
        elif menu_nb == 3: 
            class_counts = fct.count_class_occurrences(y_train) 
            label_names = [b'airplane', b'auto', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
            fct.plot_class_distribution(class_counts, label_names) 
        
        elif menu_nb == 4:
            print("creating vgg layers")
            vgg1 = dnn.create_vgg_like_architecture(num_blocks=1)
            vgg2 = dnn.create_vgg_like_architecture(num_blocks=2)
            vgg3 = dnn.create_vgg_like_architecture(num_blocks=3)
            
            print("compiling the vgg blocks")
            vgg1.compile(loss='categorical_crossentropy',optimizer= RMSprop(learning_rate=0.001), metrics=['accuracy'])
            vgg2.compile(loss='categorical_crossentropy',optimizer= RMSprop(learning_rate=0.001), metrics=['accuracy'])
            vgg3.compile(loss='categorical_crossentropy',optimizer= RMSprop(learning_rate=0.001), metrics=['accuracy'])

            for model, name in [(vgg1, "VGG1"), (vgg2, "VGG2"), (vgg3, "VGG3")]:
                history = dnn.train_model(model, x_train, y_train, x_test, y_test)
                dnn.plot_history(history, name)
                dnn.evaluate_model(model, x_test, y_test, name)

        elif menu_nb == 5:
            (x_train2, y_train2), (x_test2, y_test2) = fct.load_cifar10_dataset(data_dir)
            print("starting lenet5 algo")   
            x_train2 = x_train2.astype('float32') / 255.0  
            x_test2 = x_test2.astype('float32') / 255.0   
            
            x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train2, y_train2, test_size=0.2, random_state=42)
            print("Validation set shape (images):", x_val2.shape)  
            print("Validation set shape (labels):", y_val2.shape)

            model_lenet5 = lenet5.create_lenet5(input_shape=(32, 32, 3), num_classes=10)
            model_lenet5.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.001) , metrics=['accuracy'])

            history_lenet5 = model_lenet5.fit(x_train2, y_train2,                     
                                   validation_data=(x_val2, y_val2), 
                                   epochs=10, 
                                   batch_size=32)

            print("LeNet5 Model Training Complete")
            lenet5.evaluate_and_plot(model_lenet5, x_test2, y_test2, history_lenet5, "LeNet-5")


        elif menu_nb == 0:  
            print("Exiting the menu.")
            break 

        else:
            print("Invalid choice. Please choose a valid option.")

if __name__ == "__main__":
    main()