import asyncio
import numpy as np
import pandas as pd
from backend import load_model, preprocess_image_bytes, compute_shap_importance, CLASS_ID_TO_NAME

model = load_model()

with open("/Users/kiran/Downloads/fyp/transformed_AneRBC_dataset/transformed_AneRBC-II/Microcytic/transformed_Original_images/0001_01_Microcytic.png", "rb") as f:
    img_bytes = f.read()
image_np = preprocess_image_bytes(img_bytes)

df = pd.read_csv("/Users/kiran/Downloads/fyp/transformed_AneRBC_dataset/transformed_AneRBC-II/Microcytic/transformed_CBC_reports/0001_01_Microcytic.csv")
CBC_FEATURES = ["WBC","RBC","HGB","HCT","MCV","MCH","MCHC","PLT","MPV","RDW_CV"]
TRAIN_MEDIANS = {
    "WBC":7.94,"RBC":4.70,"HGB":10.70,"HCT":33.60,
    "MCV":75.70,"MCH":24.65,"MCHC":32.15,"PLT":293.00,
    "MPV":10.90,"RDW_CV":14.60,
}
cbc_values = {}
row = df.iloc[0]
for feat in CBC_FEATURES:
    v = row.get(feat, np.nan)
    if pd.notna(v):
        cbc_values[feat] = float(v)

for feat in CBC_FEATURES:
    cbc_values.setdefault(feat, TRAIN_MEDIANS[feat])

cbc_np = np.array([cbc_values[f] for f in CBC_FEATURES], dtype=np.float32)

inputs = {"image_input": np.expand_dims(image_np, 0), "cbc_input": np.expand_dims(cbc_np, 0)}
probs = model.predict(inputs, verbose=0)[0]
pred_idx = int(np.argmax(probs))
print("Prediction:", CLASS_ID_TO_NAME[pred_idx], probs)

shap_res = compute_shap_importance(model, cbc_np, image_np, pred_idx)
print(shap_res)
