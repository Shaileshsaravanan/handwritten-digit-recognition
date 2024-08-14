from keras.models import model_from_json
import cv2

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
image_test = cv2.imread('dataset/6c.jpg', 0)
image_test_1 = cv2.resize(image_test, (28,28))   
image_test_2 = image_test_1.reshape(1,1,28,28)   
loaded_model_pred = loaded_model.predict_classes(image_test_2, verbose = 0)
print('Prediction of loaded_model: {}'.format(loaded_model_pred[0]))