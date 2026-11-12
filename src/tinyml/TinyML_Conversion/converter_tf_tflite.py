
import pandas as pd
import tensorflow as tf
import numpy as np

X_calibration= np.load("calibration_data.npy").astype(np.float32)
def representative_data_gen():
    # Provide ~100 to 200 samples to calibrate the scale and zero_point
    for i in range(720):
        # Must match the shape of the dummy_input from Step 1: (1, 67)
        sample = X_calibration[i].reshape(1, 67)
        yield [sample]

def convert_to_tflite(saved_model_dir, output_filename):
    print(f"Converting {saved_model_dir} to {output_filename}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Enforce INT8 Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Force Inputs and Outputs to be INT8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved {output_filename}!")

# Run the conversions
convert_to_tflite("saved_model", "model_quant.tflite")
convert_to_tflite("saved_model_enc", "model_enc_quant.tflite")