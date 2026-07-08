# CROSS-BRANCH INTERACTION FUSION WITH HUMAN-OBJECT TRANSFORMER (CBIF-HOTR) FOR VIOLENCE RECOGNITION


<p align="center">
  <img src="./images/CBIF-HOTR architecture.png" width="700">
</p>

## Short description (for README intro / social preview)

> **CBIF-HOTR** is an explainable violence detection framework that reasons jointly over Human-Object Interactions (HOI) and Human-Human Interactions (HHI). Built on the HOI Transformer (HOTR), it adds a dedicated HHI branch to model aggressor-victim relationships and a Confidence-Gated Bidirectional Cross-Attention Fusion (CBAF) module to let the two branches exchange information while keeping their task-specific representations intact. Instead of a single scene-level label, the model outputs structured interaction triplets covering object interactions, human-human interactions, weapon usage, and violence-related actions — giving surveillance systems interpretable, evidence-backed explanations rather than a black-box "violent/non-violent" score.




Model Download Link: [CBIF_HOTR](https://drive.google.com/file/d/1q1HXf-jx5IbemV2CuQ7AApiSArN0X11u/view?usp=drive_link)


Sample Output
<p align="center">
  <img src="./images/HHI_HOI_triplet.png" width="700">
</p>


## Requirements

- Python 3.10 and above
- See `requirement.txt` for the full list of dependencies

## Installation

```bash
# Clone the repository
git clone [https://github.com/username/project-name.git](https://github.com/Lscorpy/CROSS-BRANCH-INTERACTION-FUSION-WITH-HUMAN-OBJECT-TRANSFORMER.git)
cd CBIF_HOTR

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt
```

## Usage for train

```bash
python Run_CBIF_HOTR.py
```

## Usage for test result
download the model from Google Drive
```bash
python visualize_three_stage.py
```

## Project Structure

```
CBIF_HOTR/
├── dataset/               # Raw and processed data
├── CBIF_HOTR/              # Saved model checkpoints
├── checkpoint/           
├── Run_CBIF_HOTR.py                 # Source code
├── requirement.txt       # Python dependencies
└── README.md
```



```markdown
## Acknowledgements

This project is built upon [Official repository for HOTR: End-to-End Human-Object Interaction Detection with Transformers (CVPR'21, Oral Presentation)](https://github.com/kakaobrain/hotr.git), 

originally developed by [bmsookim]. We thank the original authors for making their code available.

Modifications in this repository include:
- Added a Human-Human Interaction (HHI) branch
- Introduced the Confidence-Gated Bidirectional Cross-Attention Fusion (CBAF) module
- Changes in Matcher, criterition
- Add Validation Method
```

Original HOTR work:

```bibtex
@INPROCEEDINGS {9578076,
author = { Kim, Bumsoo and Lee, Junhyun and Kang, Jaewoo and Kim, Eun-Sol and Kim, Hyunwoo J. },
booktitle = { 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) },
title = {{ HOTR: End-to-End Human-Object Interaction Detection with Transformers }},
year = {2021},
volume = {},
ISSN = {},
pages = {74-83},
doi = {10.1109/CVPR46437.2021.00014},
url = {https://doi.ieeecomputersociety.org/10.1109/CVPR46437.2021.00014},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =Jun}

```

