from ae_loader import extract_dt_features
from tree_loader import predict_tree


def predict_ae_dt(features):

    dt_features = extract_dt_features(features)

    return predict_tree(dt_features)