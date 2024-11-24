# CLAD

This repository contains code and utilities for Contrastive Learning and Adversarial Disentanglement for Task Oriented Communications - https://arxiv.org/abs/2410.22784

Please note, the repo will be updated for improved functionality, usage, and documentation by 25th December, 2024. Thank you for your patience. 
## Requirements

To run the code in this repository, you need the following:

- Python 3.8 or higher
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/OmarErak/CLAD.git
    cd CLAD
    ```

2. **Create a virtual environment:**
    ```sh
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Open the Jupyter Notebook:**
    ```sh
    jupyter notebook
    ```

2. **Navigate to the `CLAD_Colored_FashionMNIST\Final_Contrastive_V2.ipynb` notebook and open it.**

3. **Run the notebook cells:**


## File Structure

- **auto_augment.py**: Implements data augmentation techniques used to expand the training dataset and improve model robustness.
- **duplicate_sample_transform.py**: Contains utilities for identifying and transforming duplicate samples.
- **utils.py**: General helper functions used across the project.
- **cnn_contrastive.py**: Contains the CNN architecture and contrastive learning functions.
- **spc.py**: Implements supervised contrastive loss.
- **Final_Contrastive_V2.ipynb**: Jupyter Notebook for training and testing CLAD.

## TODO
- Add CIFAR10 and ColoredMNIST folders
- Modular code instead of notebooks
- Add baseline implementations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
If you find the paper or codebase useful, please cite our paper: 
~~~
@misc{erak2024contrastivelearningadversarialdisentanglement,
      title={Contrastive Learning and Adversarial Disentanglement for Privacy-Preserving Task-Oriented Semantic Communications}, 
      author={Omar Erak and Omar Alhussein and Wen Tong},
      year={2024},
      eprint={2410.22784},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.22784}, 
}
~~~
