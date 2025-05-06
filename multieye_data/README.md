# MultiEYE Dataset

## Data Acquisition

MultiEYE is constructed by assembling 12 public fundus datasets and 4 OCT datasets with our private data. Our dataset is released at []. After downloading the data, please place them in this folder, following the structure outlined below.

## Data Structure Overview

```
.
├── assemble
├── |-- train
├── |   |-- ImageData
├── |   `-- large9cls.txt
├── |-- dev
├── |   `-- large9cls.txt
├── `-- test
│   └── `-- large9cls.txt
├── assemble_oct
├── |-- train
├── |   |-- ImageData
├── |   `-- large9cls.txt
├── |-- dev
├── |   `-- large9cls.txt
└── `-- test
    └── `-- large9cls.txt
```

## Data Organization and Labeling

`assemble` contains fundus data, and `assemble_oct` contains OCT data.

For each modality, all images are stored in the `ImageData` folder.

The file `large9cls.txt` contains the labels for each subset. The labels are integers ranging from 0 to 8, each corresponding to a specific eye condition or normal status: 

- 0: Normal
- 1: Dry Age-Related Macular Degeneration (dAMD)
- 2: Central Serous Chorioretinopathy (CSC)
- 3: Diabetic Retinopathy (DR)
- 4: Glaucoma (GLC)
- 5: Macular Epiretinal Membrane (MEM)
- 6: Retinal Vein Occlusion (RVO)
- 7: Wet Age-Related Macular Degeneration (wAMD)



