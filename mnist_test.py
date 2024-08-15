import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential

def model():
    # Load the JSON model
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    # Load model using custom_objects
    loaded_model = model_from_json(loaded_model_json, custom_objects={"Sequential": Sequential})
    
    # Load weights into the model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    
    # Compile the model (optional, if needed)
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return loaded_model