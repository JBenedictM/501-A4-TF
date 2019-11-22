import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import os

def main():
     class_names = check_args()
     
     print("--Load Model {%s}--" %sys.argv[2])
     #Load the model that should be in sys.argv[2]
     loaded_model = tf.keras.models.load_model(sys.argv[2])
     loaded_model.summary()
     
     print("--Load Folder {%s}--" %sys.argv[3])
     imgs = os.listdir(sys.argv[3])
     for img_filename in imgs:
          actual_path = sys.argv[3] + img_filename
          print("--Test Image {%s}--" %actual_path)
          # open image source folder
          img = plt.imread(actual_path)
          #print(img)
          
          if np.amax(img.flatten()) > 1:
               print("normalizing_data")
               img = img/255

          img = 1 - img
          # parse image name to get actual value
          img_name_parsed = img_filename.split("_")
          true_label = -1
          for word in img_name_parsed:
               if (word[0] == "a"):
                    true_label = int(word[2])
                    break
               
          print("--Predict as Class {%s}--" %(class_names[true_label]))
          
          predict(loaded_model, class_names, img, true_label)
          #break

def predict(model, class_names, img, true_label):
    img = np.array([img])
    img2 = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1)

    #Replace these two lines with code to make a prediction
    prediction = model.predict(img2)[0]
    #Determine what the predicted label is
    predicted_label = np.argmax(prediction)
    plot(class_names, prediction, true_label, predicted_label, img[0])
    plt.show()

def check_args():
     if(len(sys.argv) != 4):
          print("Usage python predict.py <MNIST,notMNIST> <model.h5> <folder path of images>")
          sys.exit(1)
     if sys.argv[1] == "MNIST":
          print("--Dataset MNIST--")
          class_names = list(range(10))
     elif sys.argv[1] == "notMNIST":
          print("--Dataset notMNIST--")
          class_names = ["A","B","C","D","E","F","G","H","I","J"]
     else:
          print(f"Choose MNIST or notMNIST, not {sys.argv[1]}")
          sys.exit(2)
     if sys.argv[2][-3:] != ".h5":
          print(f"{sys.argv[2]} is not a h5 extension")
          sys.exit(3)
     if not os.path.exists(sys.argv[3]):
          print(f"{sys.argv[3]} is not a valid folder")
          sys.exit(3)
          
     #img = plt.imread(sys.argv[3])
     #if len(img.shape) != 2:
     #     print("Image is not grey scale!")
     #     sys.exit(4)
     #if img.shape != (28,28):
     #     print("Image is not 28 by 28!")
     #     sys.exit(4)
     
     return class_names

def plot(class_names, prediction, true_label, predicted_label, img):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(prediction),class_names[true_label]),color=color)
    plt.subplot(1,2,2)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(class_names, prediction, color="#777777")
    plt.ylim([0, 1])
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == "__main__":
    main()
