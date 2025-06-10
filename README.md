## GCNet: a Combination of Graph and Convolutional Neural Network for Texts

## Introduction

This project is part of [GCNet](). This project explores the use of deep learning techniques, specifically the combination of Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs), for tasks on texts such as text classification and text embedding. Here is some of use-cases where explored or want to explore:

- Text classification
- Spam detection
- Search engine optimization (SEO)
- Vector embedding for RAG and Vector database
- Knowledge distillation
- Named entity recognition (NER)

## Project Structure

- **datasets/**: Contains datasets used for training and evaluation.
- **FindBestModel/**: Contains Jupyter notebooks for testing and identifying the best model.
- **TestsOnGrambedding/**: Contains Jupyter notebooks for testing and comparing the model against state-of-the-art models using the Grambedding dataset.
- **TestsOnMaliciousURLs/**: Contains Jupyter notebooks for testing and comparing the model against state-of-the-art models using the Malicious URLs dataset.
- **TestsOnPhishStorm/**: Contains Jupyter notebooks for testing and comparing the model against state-of-the-art models using the PhishStorm dataset.
- **TestsOnSpamURLs/**: Contains Jupyter notebooks for testing and comparing the model against state-of-the-art models using the Spam dataset.
- **Visualizations/**: Python scripts for training, evaluation, and preprocessing.
- **Visualizations/evaluations**: Python scripts for training, evaluation, and preprocessing.
- **Visualizations/visualization_records**: Python scripts for training, evaluation, and preprocessing.

## Model Architecture

<img alt="The model architecture" src="ModelArchitecture.jpg">

## Dependencies

**_requirements.txt_**: Use this file to install required packages.

## Citation

```bibtext
@article{rastakhiz2024quick,
  title={QuickCharNet: An Efficient URL Classification Framework for Enhanced Search Engine Optimization},
  author={Rastakhiz, Fardin and Eftekhari, Mahdi and Vahdati, Sahar},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```

## License

For detailed licensing information, please see the [LICENSE](LICENSE) file.

## a

$$
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}_{s}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)}
        \alpha_{i,j}\mathbf{\Theta}_{t}\mathbf{x}_{j},
$$

$$
\alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
        \right)\right)}
$$

### New Formula:

$$
        \mathbf{x}^{\prime}_{k,i} = \alpha_{k,i,i}\mathbf{\Theta}_{k,s}\mathbf{x}_{k,i} +
        \sum_{j \in \mathcal{N}(k,i)}
        \alpha_{k,i,j}\mathbf{\Theta}_{k,t}\mathbf{x}_{k,j},
$$

$$
\alpha_{k,i,j} =
        \frac{
        \exp\left(\mathbf{a}_{k}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{k,s} \mathbf{x}_{k,i} + \mathbf{\Theta}_{k,t} \mathbf{x}_{k,j}
        \right)\right)}
        {\sum_{l \in \mathcal{N}(k,i) \cup \{ i \}}
        \exp\left(\mathbf{a}_{k}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{k,s} \mathbf{x}_{k,i} + \mathbf{\Theta}_{k,t} \mathbf{x}_{k,l}
        \right)\right)}
$$

Where $k$ is the number of heads
