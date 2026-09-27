# Task: The CIFAR-100 Architecture Shootout & Unfreezing Challenge

**Your Mission:**
You will apply Transfer Learning to the CIFAR-100 dataset (100 categories of objects).

**Deadline:**
Thursday the 3rd of the month at 11:59 PM

---

**Step-by-Step Instructions:**

**Part 1: The Model Shootout (Feature Extraction)**

* **Model A:** Build a Transfer Learning model using `ResNet50V2` as a frozen backbone.
* **Model B:** Build a Transfer Learning model using `MobileNetV2` as a frozen backbone.
* Train both on CIFAR-100 for **5 epochs** each using Feature Extraction (`base_model.trainable = False`).
* **Compare:** Which model achieved higher validation accuracy? Which one trained faster?

**Part 2: The Winner's Circle (Percentage-Based Unfreezing)**
Take the winning model from Part 1, set `base_model.trainable = True`, and test two unfreezing depths based on the total number of layers:

* **Strategy A (Top 10% Unfreezing):**
* If `ResNet50V2` wins (190 total layers): Unfreeze the last 19 layers.
* If `MobileNetV2` wins (154 total layers): Unfreeze the last 15 layers.


* **Strategy B (Deep 35% Unfreezing):**
* If `ResNet50V2` wins (190 total layers): Unfreeze the last 65 layers.
* If `MobileNetV2` wins (154 total layers): Unfreeze the last 54 layers.



**How to implement the percentage split dynamically in Python:**

```python
# 1. Unfreeze the base model
base_model.trainable = True

# 2. Calculate the split index for your target unfreeze percentage (e.g., 0.10 or 0.35)
total_layers = len(base_model.layers)
freeze_until = int(total_layers * (1.0 - unfreeze_percentage))

# 3. Freeze all layers before the split point
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

```

* Fine-tune each strategy for **5 epochs** with `learning_rate = 1e-5`.
* **Analyze:** Which unfreezing depth gave the best final accuracy? Did deep unfreezing cause any overfitting?

---

**Helper Notes for CIFAR-100:**

* CIFAR-100 images are 32x32. Pre-trained backbones expect larger images, so add a `layers.Resizing(128, 128)` layer.
* `ResNet50V2` and `MobileNetV2` expect inputs in the range [-1, 1], so add `layers.Rescaling(1./127.5, offset=-1)`.
* Keep Batch Normalization running in inference mode during fine-tuning by calling `base_model(inputs, training=False)`.