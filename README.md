# [AAAI 2026] Leveraging Textual Compositional Reasoning for Robust Change Captioning
Official Repository for "Leveraging Textual Compositional Reasoning for Robust Change Captioning" (AAAI 2026)

[![CORTEX paper](https://img.shields.io/badge/CORTEX-paper-brightgreen)](https://www.arxiv.org/abs/2511.22903)


## Installation

1. Install requirements:
   ```
    git clone https://github.com/VisualAIKHU/CORTEX.git
    cd CORTEX

    conda create -n cortex python=3.8
    conda activate cortex

    pip install -r requirements.txt
   ```

2. Setup COCO caption eval tools ([github](https://github.com/tylin/coco-caption)).

## Data Preparation

### 1. Datasets
1) CLEVR-Change ([data](https://github.com/Seth-Park/RobustChangeCaptioning?tab=readme-ov-file)) 
2) CLEVR-DC ([data](https://github.com/hsgkim/clevr-dc?tab=readme-ov-file))
3) Spot-the-Diff ([data](https://github.com/harsh19/spot-the-diff?tab=readme-ov-file))


- Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128

# processing distractor images
python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
```

- Build vocab and label files using caption annotations (files are already provided in `data/`):
```
python scripts/preprocess_captions_transformer.py --input_captions_json ./data/change_captions.json --input_neg_captions_json ./data/no_change_captions.json --input_image_dir ./data/images --split_json ./data/splits.json --output_vocab_json ./data/transformer_vocab.json --output_h5 ./data/transformer_labels.h5
```


### 2. RTE Datasets
We propose an RTE module that leverages a VLM to extract structured textual cues that encode explicit compositional reasoning elements from images.

The raw RTE files are located in `data/RTE_datasets`, and the embedded RTE features are generated using `scripts/RTEtoh5.py` (already provided as `data/BefCap_chg.h5` and `data/AftCap_chg.h5`).


### 3. Directory Structure:
```
└── data
    ├── images
    ├── sc_images
    ├── nsc_images
    ├── features
    │   └── CLEVR_default_000000.png.npy
    │
    ├── sc_features
    │   └── CLEVR_semantic_000000.png.npy
    │
    ├── nsc_features
    │   └── CLEVR_nonsemantic_000000.png.npy
    │    
    ├── RTE_datasets
    │   ├── CLEVR-Change
    │   │   ├── aft.json
    │   │   └── bef.json
    │   ├── CLEVR-DC
    │   └── Spot-the-Diff
    │
    ├── BefCap_chg.h5
    ├── AftCap_chg.h5
    ├── change_captions.json
    ├── no_change_captions.json
    ├── splits.json
    ├── total_change_captions_reformat.json
    ├── transformer_vocab.json
    └── type_mapping.json
```


## Usage

To train, test, and evaluate the CORTEX model:

```bash
bash run.sh
```

## Model Checkpoint
We provide the model checkpoints for CORTEX (DIRL). [checkpoint]()


## Acknowledgements

Our codes benefits from the excellent [DIRL](https://github.com/tuyunbin/DIRL), [SMART](https://github.com/tuyunbin/SMART), [SCORER](https://github.com/tuyunbin/SCORER).


## Citation
```
@article{park2025leveraging,
  title={Leveraging Textual Compositional Reasoning for Robust Change Captioning},
  author={Park, Kyu Ri and Park, Jiyoung and Kim, Seong Tae and Lee, Hong Joo and Kim, Jung Uk},
  journal={arXiv preprint arXiv:2511.22903},
  year={2025}
}
```
