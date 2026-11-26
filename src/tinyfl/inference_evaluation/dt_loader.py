import ctypes
import numpy as np

tree = ctypes.CDLL("./tree_model.so")

tree.my_model_predict.argtypes = [
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int32
]

tree.my_model_predict.restype = ctypes.c_int32


def predict_tree(features):

    features = np.ascontiguousarray(
        features,
        dtype=np.int16
    ).reshape(14)

    return tree.my_model_predict(
        features.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16)
        ),
        len(features)
    )