# Cats vs Dogs Transfer Learning Project

This repository showcases how to adapt a **pre‑trained convolutional neural network (CNN)** to classify images of cats and dogs using transfer learning. The model leverages **MobileNetV2**, trained on ImageNet, and fine‑tunes it on the *cats_vs_dogs* dataset.

## Files

- `final_assignment.ipynb` – An English Jupyter notebook that implements the entire workflow: data loading, preprocessing and augmentation, model creation, feature‑extraction training, fine‑tuning, evaluation, and model saving. The notebook uses TensorFlow and TensorFlow Datasets.
- `project_documentation_en.md` – A detailed English documentation file explaining the dataset, model architecture, training procedure and methodology.
- `project_documentation.md` – Arabic documentation (useful if you need the project in Arabic; otherwise optional).

## Requirements

- **Python** 3.10 or newer
- **TensorFlow** 2.x
- **TensorFlow Datasets**
- **Matplotlib** (for plotting)

To install the necessary libraries on your local machine, run:

```bash
pip install tensorflow tensorflow-datasets matplotlib
```

On Google Colab these dependencies are already available.

## Usage

1. Clone or download this repository.
2. Open `final_assignment.ipynb` in Jupyter Notebook or upload it to Google Colab.
3. Run all cells in order. The notebook will:
   - Download and load the *cats_vs_dogs* dataset (23,262 valid training examples after dropping corrupted images).
   - Resize images to `224×224`, apply data augmentation and prepare training/validation splits.
   - Load MobileNetV2 without its top and freeze its weights.
   - Add a new classification head and train it on the dataset (feature extraction).
   - Unfreeze some of the higher layers and fine‑tune the network with a low learning rate.
   - Plot training and validation accuracy, evaluate the model and save the fine‑tuned model as `cats_dogs_finetuned.h5`.

Running the notebook for the first time will automatically download the dataset via TensorFlow Datasets. Subsequent runs will use the cached copy.

## References

- Keras guide on transfer learning and fine‑tuning: https://keras.io/guides/transfer_learning/
- TensorFlow Datasets catalog entry for *cats_vs_dogs*: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs

---

Feel free to modify the notebook to experiment with different pre‑trained models (e.g. ResNet50, EfficientNetB0) or different fine‑tuning depths. Contributions are welcome!