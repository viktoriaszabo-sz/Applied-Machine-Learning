import os
from keras.applications import Xception
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

# Load the pretrained model from internet/computer (approximately 88 MB)
pretrained_model = Xception(weights="imagenet")

# Directory containing the images
folder_path = 'Fruties/images/'

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
        img_path = os.path.join(folder_path, file_name)
        img = load_img(img_path, target_size=(299, 299))  # Resize the image directly when loading

        # Convert the image to array
        img_array = img_to_array(img)

        # Convert the image into a 4 dimensional Tensor
        # Convert from (height, width, channels) to (batchsize, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the input image array
        img_array = imagenet_utils.preprocess_input(img_array, mode='tf')  # Ensure mode is set to 'tf' for Xception

        # Predict using predict() method
        prediction = pretrained_model.predict(img_array)

        # Decode the prediction
        actual_prediction = imagenet_utils.decode_predictions(prediction, top=1)[0]

        # Print predicted object
        print(f"File: {file_name}")
        print("Predicted object is:")
        print(actual_prediction[0][1])  # label
        print("With accuracy")
        print(actual_prediction[0][2] * 100)  # probability

        # Display image and the prediction text over it
        disp_img = cv2.imread(img_path)
        cv2.putText(disp_img, f"{actual_prediction[0][1]}: {actual_prediction[0][2]*100:.2f}%", (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0))
        cv2.imshow("Prediction", disp_img)
        cv2.waitKey(0)  # Press any key to close the image window

cv2.destroyAllWindows()
