import xgboost as xgb
import numpy as np
from aiml.xgboost.main_data3 import classify_model


def inference(model, df_outbag, args):
    # calculate out-of-bag properties
    (y1_pred, s1), (y2_pred, s2) = classify_model(df_outbag, model, args)
    (m1, t1), (m2, t2) = model
    return y1_pred, y2_pred, s1, s2, np.array([t1]), np.array([t2])

