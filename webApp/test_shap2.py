import numpy as np
from backend import load_model, preprocess_image_bytes, CLASS_ID_TO_NAME

model = load_model()
with open("/Users/kiran/Downloads/fyp/transformed_AneRBC_dataset/transformed_AneRBC-II/Microcytic/transformed_Original_images/0001_01_Microcytic.png", "rb") as f:
    img_bytes = f.read()
image_np = preprocess_image_bytes(img_bytes)

cbc_np = np.array([8.49, 4.37, 10.1, 29.9, 68.4, 23.1, 33.8, 376.0, 11.2, 14.6], dtype=np.float32)

def predict_fn(cbc_samples):
    n = len(cbc_samples)
    imgs = np.tile(np.expand_dims(image_np, 0), (n, 1, 1, 1)).astype(np.float32)
    preds = model.predict(
        {"image_input": imgs, "cbc_input": cbc_samples.astype(np.float32)},
        verbose=0, batch_size=8
    )
    return preds[:, 1] # class 1

TRAIN_MEDIANS = {
    "WBC":7.94,"RBC":4.70,"HGB":10.70,"HCT":33.60,
    "MCV":75.70,"MCH":24.65,"MCHC":32.15,"PLT":293.00,
    "MPV":10.90,"RDW_CV":14.60,
}
CBC_FEATURES = ["WBC","RBC","HGB","HCT","MCV","MCH","MCHC","PLT","MPV","RDW_CV"]
background = np.array([[TRAIN_MEDIANS[f] for f in CBC_FEATURES]], dtype=np.float32)

import shap
explainer = shap.KernelExplainer(predict_fn, background)
print("background output:", predict_fn(background))
print("sample output:", predict_fn(cbc_np.reshape(1, -1)))
sv = explainer.shap_values(cbc_np.reshape(1, -1), nsamples=64, silent=True)
print("SHAP values:", sv)

print("sum:", np.abs(sv[0]).sum())
print("exact sv:", repr(sv))
