# Transfer Learning for Cats vs Dogs Classification

## Introduction

This project demonstrates how to adapt a **pre‑trained convolutional neural network (CNN)** to a new image classification task using transfer learning. Instead of training a deep network from scratch, we reuse a model trained on a large corpus (ImageNet) and fine‑tune it for binary classification (cats vs dogs). Transfer learning takes advantage of feature extractors learned on massive datasets, making it particularly effective when you have limited training examples.

## Dataset

We use the `cats_vs_dogs` dataset from **TensorFlow Datasets (TFDS)**. The TFDS catalog notes that the dataset contains **1,738 corrupted images that are dropped**, and the `train` split contains **23,262** valid examples. Each image is resized to `224 × 224` pixels to match the input requirements of our chosen base model.

The dataset is divided into two splits in this project:

- **Training set:** 80 % of the images (≈18 610 examples)
- **Validation set:** 20 % of the images (≈4 652 examples)

We apply data augmentation (random flipping, rotation and zoom) to the training set to reduce overfitting.

## Model Architecture

### Base model

We select **MobileNetV2**, a lightweight CNN architecture pre‑trained on ImageNet, as the base model. MobileNetV2 is known for its efficiency and strong performance on mobile and embedded devices. We load it without the original classification head (`include_top=False`), retaining only the convolutional base. Following common transfer‑learning practice, we freeze the convolutional base during the first training stage and train only the new classifier head.

### New classification head

On top of the frozen base, we add:

1. **GlobalAveragePooling2D** – converts the convolutional feature maps into a single feature vector.
2. **Dropout (0.2)** – helps mitigate overfitting by randomly deactivating neurons during training.
3. **Dense layer with one unit and sigmoid activation** – outputs a probability for the “dog” class (the other class corresponds to “cat”).

This head is randomly initialized and trained on our dataset.

## Training Procedure

### Feature extraction stage

We first freeze all layers of the MobileNetV2 base and train only the new classification head. The model is compiled with the **binary cross‑entropy** loss and the **Adam** optimizer using a learning rate of 1 × 10⁻³. Training runs for several epochs while monitoring validation accuracy.

### Fine‑tuning stage

After the head converges, we unfreeze the top portion of the base model (the last 100 layers) and continue training with a much lower learning rate (1 × 10⁻⁵). Fine‑tuning should be done with a low learning rate to avoid large updates that can harm the pre‑trained weights. We recompile the model after changing the `trainable` attributes. Fine‑tuning allows the network to adapt higher‑level features to nuances in the cats vs dogs dataset.

## Implementation Outline

The accompanying Jupyter notebook (provided separately) follows these steps:

1. **Setup:** Import TensorFlow, TFDS and Keras APIs; verify GPU availability.
2. **Data loading:** Load and split the `cats_vs_dogs` dataset from TFDS.
3. **Preprocessing:** Resize images to `224 × 224` and normalize pixel values; define data augmentation pipeline.
4. **Model creation:** Load MobileNetV2 without its top, freeze its layers, and build the new classification head.
5. **Stage 1 training:** Compile and train only the head.
6. **Stage 2 fine‑tuning:** Unfreeze selected layers, compile with a lower learning rate and continue training.
7. **Evaluation:** Plot training/validation accuracy curves and evaluate the model on the validation set.
8. **Prediction:** Demonstrate inference on a single validation image.
9. **Saving:** Save the final fine‑tuned model to an H5 file.

## Results

Although specific performance metrics depend on training conditions (number of epochs, random seed, etc.), transfer learning with MobileNetV2 usually achieves strong accuracy after only a few epochs. Fine‑tuning often boosts performance further by adapting the highest layers to the new data distribution. Users are encouraged to run the provided notebook in Google Colab and experiment with different fine‑tuning depths and hyperparameters.

## Conclusion

This project illustrates how to efficiently repurpose a pre‑trained CNN for a new classification task using transfer learning and fine‑tuning. By freezing the majority of the pre‑trained layers and training only a small head, we achieve strong baseline performance with minimal computational cost. Fine‑tuning further improves accuracy by allowing the model to refine higher‑level features. The same methodology can be extended to other pre‑trained architectures (e.g., VGG, ResNet) or to multi‑class problems by adjusting the output layer and loss function.

## References

- TensorFlow Datasets — `cats_vs_dogs`: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
- Keras guide — Transfer learning & fine‑tuning: https://keras.io/guides/transfer_learning/