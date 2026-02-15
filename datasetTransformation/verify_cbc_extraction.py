#!/usr/bin/env python3
"""Quick verification script for CBC extraction results."""

import pandas as pd

# Load CSV
csv_path = '/Users/kiran/Downloads/fyp/Code/Fusion_Model/transformedDataset/AneRBC_II_Full_Clinical_Data.csv'
df = pd.read_csv(csv_path)

print('=' * 70)
print('CBC DATA EXTRACTION VERIFICATION SUMMARY')
print('=' * 70)

print(f'\n1. File Structure:')
print(f'   Total rows: {len(df)}')
print(f'   Total columns: {len(df.columns)}')

print(f'\n2. Cohort Distribution:')
print(df['Cohort'].value_counts().to_string())

print(f'\n3. Anemia Classification:')
category_names = {
    0: 'Class 0 (Healthy)',
    1: 'Class 1 (Microcytic, MCV < 80)',
    2: 'Class 2 (Normocytic, 80 ≤ MCV ≤ 100)',
    3: 'Class 3 (Macrocytic, MCV > 100)'
}
counts = df['Anemia_Category'].value_counts().sort_index()
for cat, count in counts.items():
    print(f'   {category_names.get(cat, f"Class {cat}")}: {count}')

print(f'\n4. Data Completeness:')
cbc_params = ['WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'MPV', 'RDW_CV']
for param in cbc_params:
    missing = df[param].isna().sum()
    pct = (missing / len(df)) * 100
    print(f'   {param:<10} Missing: {missing:>3} ({pct:>5.1f}%)')

print(f'\n5. Abnormal Flags Summary:')
abnormal_cols = [col for col in df.columns if col.endswith('_abnormal')]
total_abnormal = df[abnormal_cols].sum().sum()
print(f'   Total abnormal values flagged: {int(total_abnormal)}')
print(f'   Average abnormal flags per file: {total_abnormal / len(df):.2f}')

print(f'\n6. Sample Data (first 3 anemic files):')
print(df[df['Cohort'] == 'Anemic'][['File_Name', 'Cohort', 'MCV', 'Anemia_Category']].head(3).to_string(index=False))

print('\n' + '=' * 70)
print('✓ Extraction completed successfully!')
print('=' * 70)
