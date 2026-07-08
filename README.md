# CROSS-BRANCH INTERACTION FUSION WITH HUMAN-OBJECT TRANSFORMER (CBIF-HOTR) FOR VIOLENCE RECOGNITION


<p align="center">
  <img src="./images/CBIF-HOTR architecture.png" width="700">
</p>

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
git clone https://github.com/username/project-name.git
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
