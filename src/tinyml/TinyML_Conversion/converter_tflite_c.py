import os

def tflite_to_c_array(tflite_file, c_file, array_name):
    with open(tflite_file, 'rb') as f:
        data = f.read()
        
    with open(c_file, 'w') as f:
        f.write(f"// Auto-generated from {tflite_file}\n")
        f.write(f"// Size: {len(data)} bytes\n\n")
        f.write(f"#include <stdint.h>\n\n")
        f.write(f"alignas(16) const unsigned char {array_name}[] = {{\n    ")
        
        for i, byte in enumerate(data):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n    ")
                
        f.write(f"\n}};\n")
        f.write(f"const int {array_name}_len = {len(data)};\n")
    print(f"Generated {c_file}")

# Generate your C++ headers
tflite_to_c_array("model_quant.tflite", "model.h", "model_tflite")
tflite_to_c_array("model_enc_quant.tflite", "model_encoder.h", "model_enc_tflite")