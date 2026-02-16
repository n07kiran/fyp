import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset import MultimodalDataset
from utils.models import FusionModel

# Setup
TRANSFORMED = os.path.join(os.path.dirname(__file__), 'transformedDataset')
CBC_FEATURES = ['WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'MPV', 'RDW_CV']
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load data
train_df = pd.read_csv(os.path.join(TRANSFORMED, 'train_split.csv'))

print("Class distribution:")
print(train_df['final_class'].value_counts().sort_index())

# Check a few samples
dataset = MultimodalDataset(train_df.head(10), CBC_FEATURES, augment=False, project_root=PROJECT_ROOT)

for i in range(min(3, len(dataset))):
    img, cbc, label = dataset[i]
    print(f"\nSample {i}:")
    print(f"  Image: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}, has_nan={torch.isnan(img).any()}")
    print(f"  CBC: min={cbc.min():.3f}, max={cbc.max():.3f}, has_nan={torch.isnan(cbc).any()}")
    print(f"  Label: {label}")
    
# Check if preprocessed images exist
sample_path = train_df.iloc[0]['processed_image_path']
full_path = os.path.join(PROJECT_ROOT, sample_path)
print(f"\nSample image path: {sample_path}")
print(f"Full path: {full_path}")
print(f"Exists: {os.path.exists(full_path)}")

if os.path.exists(full_path):
    img_tensor = torch.load(full_path, weights_only=True)
    print(f"Raw tensor: shape={img_tensor.shape}, min={img_tensor.min():.3f}, max={img_tensor.max():.3f}, has_nan={torch.isnan(img_tensor).any()}")
