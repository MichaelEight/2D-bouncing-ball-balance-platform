import os
from keras.models import load_model
from keras.utils import plot_model

def visualize_model(model_path, output_path='model_structure.png'):
    """
    Load a Keras model from an h5 file and visualize its architecture.
    """
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Load the model
    model = load_model(model_path)

    # Visualize the model
    plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=True)
    print(f"Model visualization saved to {output_path}")

if __name__ == "__main__":
    # Replace with the path to your model file
    model_file_path = 'saved_modelsv3/model_iteration_40.h5'
    
    # Replace with your desired output path
    output_file_path = 'model_visualization.png'

    visualize_model(model_file_path, output_file_path)
