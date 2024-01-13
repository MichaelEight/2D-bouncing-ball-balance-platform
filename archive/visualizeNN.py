from keras.models import load_model
from keras.utils import plot_model

# Load the model from the h5 file
model = load_model('saved_modelsv2/model_iteration_10.h5')  # Replace 'your_model_file.h5' with your h5 file path

# Visualize the model using Keras's plot_model function
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
