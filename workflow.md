             KiTS23 CT
                 │
                 ▼
          Intensity preprocessing
                 │
                 ▼
             Resampling
                 │
                 ▼
            Training patches
                 │
                 ▼
             U-Net baseline
                 │
                 ▼
        Multi-class segmentation
                 │
       ┌─────────┼──────────┐
       ▼         ▼          ▼
    Kidney     Tumor       Cyst
       │         │          │
       └─────────┼──────────┘
                 ▼
        Volume / measurements
                 │
                 ▼
          Model evaluation