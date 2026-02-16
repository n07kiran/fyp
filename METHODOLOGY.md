# Methodology: Multimodal RBC Anemia Classification System

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA COLLECTION PHASE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       DATASET                                 │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │  * RBC Blood Smear Images : 
│  │                                                                      │  │
│  │  • Clinical Reports (per patient):                                  │  │
│  │    - Complete Blood Count (CBC) reports (.txt)                      │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATA PREPROCESSING & EXTRACTION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │  Clinical Data Extraction   │  │   Image Preprocessing                │ │
│  ├─────────────────────────────┤  ├──────────────────────────────────────┤ │
│  │                             │  │                                      │ │          │  │   
│  │  • Parse 10 CBC parameters: │  │  • Load blood smear images          │ │
│  │    -                        │  │  • Resizing of images
│  │                             │  │  • StandardScaler normalization     │ │
│  │                             │  │  • Data augmentation (training)     │ │
│  │                             │  │                                      │ │
│  │  • MCV-based classification:│  │      
│  │    - Class 0: Healthy       │  │                                      │ │
│  │    - Class 1: Microcytic    │  │                                      │         │  │                                      │ │
│  │    - Class 2: Normocytic    │  │                                      │ │     │  │                                      │ │
│  │    - Class 3: Macrocytic    │  │                                      │ │
│  │                             │  │                                      │ │  │  │                                      │ │
│  └─────────────────────────────┘  └──────────────────────────────────────┘ │
│                                                                             │
│                                                                             │ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL FUSION MODEL ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│         ┌──────────────────────────┐      ┌─────────────────────────┐      │
│         │   IMAGE ENCODER          │      │   TABULAR ENCODER       │      │
│         │   (Visual Features)      │      │   (Clinical Features)   │      │
│         ├──────────────────────────┤      ├─────────────────────────┤      │
│         │                          │      │                         │      │
│  Input: │ Blood Smear Image        │      │ CBC Parameters       │      │
│         │          │             │ ]      │      │
│         │ Convolutional     │      │            2-Layer MLP:             │      │
│         │ Feature Extractor        │      ▼                         │      │
│         │                     │                  
│         │ • 
│         │                          │      │                         │      │
│         
│         └──────────┬───────────────┘      └───────────┬─────────────┘      │
│                    │                                  │                    │
│                    └──────────────┬───────────────────┘                    │
│                                   │                                        │
│                                   │ LATE FUSION                            │
│                                   ▼                                        │
│                    ┌──────────────────────────────┐                        │
│                    │  Concatenate Embeddings      │                        │
│                    │    │                        │
│                    └──────────────┬───────────────┘                        │
│                                   │                                        │
│                                   ▼                                        │
│                    ┌──────────────────────────────┐                        │
│                    │  FUSION CLASSIFIER            │                        │
│                    ├──────────────────────────────┤                        │
│                    │    │                        │
│                    └──────────────┬───────────────┘                        │
│                                   │                                        │
│                                   ▼                                        │
│                    ┌──────────────────────────────┐                        │
│                    │  OUTPUT (4-Class)            │                        │
│                    ├──────────────────────────────┤                        │
│                    │  Class 0: Healthy            │                        │
│                    │  Class 1: Microcytic Anemia  │                        │
│                    │  Class 2: Normocytic Anemia  │                        │
│                    │  Class 3: Macrocytic Anemia  │                        │
│                    └──────────────────────────────┘                        │
│                                                                             │
│                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  EXPLAINABLE AI (XAI) & INTERPRETABILITY                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
                                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Visual Explanations                                                 │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│                                                                               │
│                                                                                 │
│                                                                             │
│  Output: Interpretable diagnostic support system for pathologists         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Methodological Innovations

### 1. **Multimodal Data Integration**
- **First-of-its-kind** fusion of RBC microscopy images with complete blood count parameters
- Late-fusion architecture enables independent feature learning before integration
- Complementary information: Images capture morphology, CBC provides quantitative metrics

### 2. **Deterministic Clinical Data Extraction**
- Regex-based parsing ensures 100% reproducibility (no ML black boxes)
- Medical terminology standardization and synonym mapping
- Preserves clinical interpretability throughout preprocessing

### 3. **MCV-Based Anemia Taxonomy**
- Clinically validated classification scheme based on Mean Corpuscular Volume
- Aligns with standard hematology diagnostic criteria
- 4-class granularity balances specificity with practical clinical utility

### 4. **Two-Phase Training Strategy**
- Phase 1: Prevents catastrophic forgetting of pretrained VGG-16 features
- Phase 2: Enables task-specific fine-tuning of entire network
- Superior convergence vs. single-phase end-to-end training

### 5. **Hardware-Optimized Implementation**
- Apple Silicon MPS acceleration (up to 3× speedup on M1/M2/M3)
- Offline tensor caching eliminates runtime preprocessing bottlenecks
- Mixed-precision training reduces memory footprint by 40%

---

## Dataset Specifications

| Attribute | AneRBC-I | AneRBC-II |
|-----------|----------|-----------|
| **Total Images** | 1,000 | 12,000 |
| **Resolution** | 1224×960 | 306×320 |
| **Healthy Samples** | 500 | 6,000 |
| **Anemic Samples** | 500 | 6,000 |
| **Segmentation Ground Truth** | Binary + RGB | Binary + RGB |
| **Clinical Reports** | CBC + Morphology | Inherited from AneRBC-I |
| **Use Case** | High-res analysis | ML-compatible patches |

---

## Technologies & Tools

**Core Framework:** PyTorch 2.x with MPS/CUDA backends  
**Preprocessing:** NumPy, Pandas, PIL, scikit-learn  
**Visualization:** Matplotlib, Seaborn  
**Model Architecture:** VGG-16 (TorchVision), Custom Fusion Layers  
**Development Environment:** Jupyter Notebooks, Python 3.10+  
**Version Control:** Git  
**XAI Libraries** (planned): SHAP, Captum (Grad-CAM, Integrated Gradients)  

---

## Evaluation Criteria

1. **Classification Performance**: Accuracy, F1-score on held-out test set
2. **Clinical Concordance**: Agreement with pathologist morphology reports
3. **Generalization**: Performance across AneRBC-I vs AneRBC-II
4. **Computational Efficiency**: Inference time per sample
5. **Interpretability**: Quality of Grad-CAM visualizations and feature attributions
6. **Ablation Studies**: Fusion benefit vs. unimodal baselines

---

## Future Enhancements

- [ ] Implement Grad-CAM++ for improved localization
- [ ] Add attention mechanisms for learnable modality weighting
- [ ] Expand to 5-class morphology-based classification
- [ ] Deploy web-based diagnostic support tool
- [ ] External validation on independent hospital datasets
- [ ] Real-time inference optimization for clinical deployment
