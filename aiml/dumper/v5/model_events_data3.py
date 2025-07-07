import xgboost as xgb
import numpy as np
from aiml.xgboost.main_event_data3 import classify_model_sklearn, classify_model


def inference(model, df_outbag, args):
    # calculate out-of-bag properties
    if args.use_xgb_sklearn_shell:
        y1_pred, s1 = classify_model_sklearn(df_outbag, model, args)
    else:
        y1_pred, s1 = classify_model(df_outbag, model, args)
    m1, t1 = model

    return y1_pred, s1, np.array([t1])
