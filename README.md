# Vulcan Detection: Architecture for Security Testing with Extensible Resourceful Intelligence and Adaptation

## Deeplearning-based Vulnerability Detection Framework

### Introduction

Vulcan Detection aims to transform the landscape of security in computing systems by providing a comprehensive framework for vulnerability detection. This cutting-edge architecture relies on deep learning methodologies, offering extensible and customizable solutions at the forefront of technology.

### Key Features

- **Extensible Resourceful Intelligence**: Incorporates AI and machine learning algorithms that can be easily extended and adapted to various security testing scenarios.
- **Rich and Diverse Data Preprocessing Mechanisms**: Including code representation methods, graph neural networks, and sequence representation methods, allowing the free combination of preprocessing steps and adjustment of the processing flow.
- **Adaptable Framework**: Tailored to meet individual requirements with built-in support for various vulnerability detection models.
- **Abundant Datasets**: Offers a wide variety of datasets to train and test the models, providing a versatile environment for experimentation.
- **Ease of Use**: Designed with the user in mind, vulcan offers an intuitive interface that makes implementing and modifying state-of-the-art (SOTA) vulnerability detection models a breeze.

### Quick Start
Getting started with vulcan Detection is simple and straightforward. Follow the installation instructions in the provided documentation, and you'll be ready to explore and customize the wide array of vulnerability detection models.
#### Required environment
- Python3 && PIP installed
- Linux/Windows/macOS
- Torch + CUDA enabled


#### Conda Installation
推荐使用 conda 保证依赖完整（Python 3.10，PyTorch/CUDA 等由 vulcan.yaml 提供）。`vulcan.yaml` 已与 `pyproject.toml` 对齐，仅 DGL 需从官方 wheel 单独安装。

**方式一：从 vulcan.yaml 创建环境（推荐）**
```bash
git clone https://github.com/NISL-Vulcan/Vulcan.git
cd Vulcan
conda env create -f vulcan.yaml -n vulcan
conda activate vulcan
# DGL 1.1.3 不在 PyPI，需从官方 wheel 安装
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install -e .
```

**方式二：仅创建 Python 3.10 环境后安装**
```bash
conda create -n vulcan python=3.10 -y
conda activate vulcan
cd Vulcan
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install -e .
```

#### Direct Installation
Type the following commands in your Shell
```
git clone https://github.com/NISL-Vulcan/Vulcan.git
cd Vulcan && chmod +x ./install.sh
./install.sh
```
#### Docker Implementation
` TODO: docker run`

### Usage Example
This framework is primarily used through configuration files written in YAML syntax. 

These configuration files contain information about the model, dataset, metrics , preprocessing, as well as configurations required for training, validation, and inference. 

You can conveniently utilize it through **Jupyter Notebook** or opt for a **command-line** method.

We've already included several notebook files in the 'notebooks' directory for your reference. 
You're welcome to delve deeper into them.

#### Jupyter Notebook
Navigate to the 'notebooks' directory and use Jupyter in the way you're accustomed to.

#### Command-Line（基于 src/vulcan 布局）
We take the ReGVD method paired with the REVEAL dataset as an example.

```bash
# Recommended installation (using pyproject.toml)
pip install -e .

# Training
vulcan-train --cfg configs/regvd_reveal.yaml

# Validation
vulcan-val --cfg configs/regvd_reveal.yaml
```

Legacy (not recommended): the old wrapper scripts under `tools/` are kept for compatibility, but the console scripts
above are the preferred entrypoints for the `src/` layout.

```bash
vulcan-train --cfg configs/regvd_reveal.yaml
vulcan-val --cfg configs/regvd_reveal.yaml
```

For more details about environment setup and usage, see `docs/usage.md`.
### Model Zoo
Supported Models/Modules:
- [Devign](https://github.com/epicosy/devign)
- [ReGVD](https://github.com/daiquocnguyen/GNN-ReGVD)
- [LineVul](https://github.com/awsm-research/LineVul/blob/main/linevul/linevul_main.py)
- VulDeePecker
- IVDetect
- LineVul
- ReGVD
- DeepWuKong
- TextCNN
- VulCNN
- transformers
- VulBERTa

### Supported Datasets
- REVEAL
- FFMPEG + Qemu
- SARD/NVD
- MSR_20_CODE/BigVul
- CODEXGLUE
- D2A
- ...

### Supported Preprocess Methods
- "Normalize"
- "PadSequence"
- "OneHotEncode"
- 'Tokenize'
- 'VocabularyMapping'
- 'LengthNormalization'
- 'Shuffle'
- ...
  
### Supported Representations Methods
- **AST Graph extractor / AST Graph builder**
- **LLVM Graph extractor / builder**
- **Syntax extractor / builder**
- **Vectorizers/word2vec**
- **sent2vec**
- ...


### Customization

vulcan Detection prides itself on its flexibility. Users can easily tailor the system to their specific needs, thanks to the comprehensive set of tools and configurations. Whether you are a seasoned security professional or new to vulnerability detection, vulcan provides an adaptable solution.

### Conclusion

vulcan Detection represents the pinnacle of vulnerability detection frameworks, offering unmatched extensibility, intelligence, and adaptability. Its ease of use, coupled with a rich selection of datasets, ensures that users can effortlessly navigate the complex terrain of secure computing.

For detailed guides, tutorials, and API references, please refer to the full documentation available in the repository.

Start exploring vulcan Detection today and take a step forward into a secure and resilient digital future.
