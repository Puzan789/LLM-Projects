
# 🇳🇵 Constitution of Nepal Q&A Model

This repository contains a fine-tuned **LLaMA-2 7B model** using **QLoRA**, based on the **Constitution of Nepal** in English. The goal of this project is to create a question-answering system that provides detailed and accurate answers from the constitution. The constitution was converted from PDF to text format, and a custom question-answer dataset was created for fine-tuning.

## Project Overview

### Key Highlights:

- **Data Source**: The English version of the Constitution of Nepal served as the primary dataset for fine-tuning.
- **Model**: Fine-tuned the **LLaMA-2 7B model** using **QLoRA** for efficient training.
- **Question Answer Dataset**: A custom dataset was generated based on the constitution’s content, containing relevant questions and answers.
- **Tools and Models**:
  - **Google Gemini Pro** was used to create the initial question-answer dataset.
  - **QLoRA (Quantized LoRA)** was applied to optimize the fine-tuning process of the LLaMA model.

## Features

- **Constitutional Q&A**: The model can answer questions about the Constitution of Nepal, making it useful for legal research, education, and general reference.
- **Efficient Fine-Tuning**: Leveraged **QLoRA** to fine-tune the model efficiently, ensuring lower memory consumption while maintaining high performance.
- **Custom Dataset**: The dataset is based on the Constitution of Nepal, ensuring high relevance and accuracy for questions related to the constitution.


## Dataset

The dataset was created by processing the English PDF version of the Constitution of Nepal and converting it into a **text format**. Using **Google Gemini Pro**, a set of relevant question-answer pairs was generated, covering key articles, provisions, and rights in the constitution.

### Explanation of Hyperparameters and Concepts for **Quantization LoRA (QLoRA)** and **PEFT** in Simple Terms:

---

### 1. **Key Hyperparameters** for Fine-Tuning Large Language Models (LLMs) Using QLoRA

Let’s break down some of the most important hyperparameters used in the fine-tuning process, especially when working with models like LLaMA-2 and applying QLoRA.

#### **LoRA Hyperparameters (Low-Rank Adaptation):**
- **`lora_r` (Attention Dimension):**  
  This parameter controls the size of the low-rank matrices used to capture changes during fine-tuning. The "rank" essentially limits how much the model adapts to the new data.  
 

- **`lora_alpha` (Scaling Factor):**  
  This scaling factor adjusts how much impact the LoRA updates have. It's like a volume control, ensuring that the changes introduced by LoRA are balanced with the original model.  
 

- **`lora_dropout` (Dropout Probability):**  
  Dropout adds a regularization technique to prevent overfitting. During training, some LoRA neurons are turned off randomly, forcing the model to be more generalizable.  


#### **Quantization Hyperparameters:**
- **`use_4bit` (4-bit Quantization):**  
  This hyperparameter tells the system to represent the model’s weights using only 4 bits. Typically, models use 16 or 32 bits, so this cuts down the memory required significantly.  
 

- **`bnb_4bit_compute_dtype` (Precision Type for Computation):**  
  Once quantized to 4 bits, the model computations may still use higher precision (such as 16-bit floats). This helps keep computations accurate, even if the storage of data is in a lower precision.  
  

- **`bnb_4bit_quant_type` (Quantization Type):**  
  Quantization types like **FP4** or **NF4** are different ways of doing 4-bit quantization. **NF4** (normal-float-4) often gives better accuracy by adjusting the distribution of the data.  
 

- **`use_nested_quant` (Nested Quantization):**  
  This is an advanced option that allows quantizing a model twice—essentially compressing it even more. It can save additional memory but might slightly reduce performance.  
 

#### **Training Parameters (General Settings for Fine-Tuning):**
- **`learning_rate`:**  
  This controls how fast the model learns. A smaller learning rate allows the model to learn slowly and steadily, which is usually better for fine-tuning large models.
- **`per_device_train_batch_size`:**  
  This specifies how many samples are processed in a single step. The larger the batch size, the faster the model can learn, but it also requires more memory.
- **`gradient_accumulation_steps`:**  
  If the batch size is too small due to memory constraints, this hyperparameter allows accumulating gradients across multiple steps to simulate a larger batch.
- **`warmup_ratio`:**  
  In the early stages of training, the learning rate starts small and gradually increases. This parameter controls how much of the training phase is dedicated to slowly warming up the learning rate.

---


## 🤔 **So, What’s the Deal with Quantization, LoRA, QLoRA, and PEFT?**  
### *Let’s Break Down These Cool Model Optimization Tricks!*

---


### 1. **Quantization**

Quantization is a technique to reduce the memory footprint of a model by representing its weights using fewer bits. Traditionally, models store each weight as a 32-bit or 16-bit floating-point number. Quantization compresses these weights into smaller representations (e.g., 4 bits), which drastically reduces memory requirements.

#### Types of Quantization:
- **FP4 (Floating Point 4-bit Quantization):**
  - In FP4, each weight of the model is stored using 4 bits, which gives 16 possible values (since 4 bits = 2<sup>4</sup> = 16 values). The quantized value is chosen from a fixed range.
  - This is a standard quantization technique that directly converts the weights from their full precision form (e.g., 32-bit) to a lower precision (4-bit).
  - Mathematically, quantization in FP4 can be represented as:
    ```
    W_quantized = Round(W_original, Δ)
    ```
    where `Δ` is the quantization step size, and `W_original` is the original 32-bit weight value.

    The quantization error, `ε`, is the difference between the original weight and the quantized weight:
    ```
    ε = W_original - W_quantized
    ```

- **NF4 (Normal Float 4-bit Quantization):**
  - **NF4** is an improved version of FP4. In NF4, the distribution of the quantized values is adapted based on the distribution of the original values, which helps reduce quantization errors.
  - In FP4, the same quantization step size `Δ` is used for all values, but in **NF4**, the quantization bins are adjusted dynamically to better match the distribution of the original data.
  - **Mathematical Explanation**:  
    - Let `W_original` be the original model weights. The **NF4** quantization assigns the values to the nearest representative from a **non-uniform set of bins** `B` derived from the statistical distribution of `W_original`.
    ```
    W_quantized = arg min_b∈B |W_original - b|
    ```
    where `B` is a set of dynamically chosen bin edges.
  
  - **Why is NF4 better?**  
    In NF4, the model learns to adjust its range of representable values based on the statistics of the weights, which allows the quantized model to better capture important weight values, resulting in less information loss compared to FP4.

### 2. **LoRA (Low-Rank Adaptation)**

LoRA focuses on efficient fine-tuning by inserting **low-rank matrices** into each layer of a model. Instead of updating the entire weight matrix `W`, only the low-rank matrices are trained.

#### Mathematical Explanation of Low-Rank Adaptation:

In a standard deep learning model, each layer has a weight matrix `W ∈ R^{d × k}`, where `d` is the number of input features, and `k` is the number of output features. When fine-tuning, instead of updating all the elements of this matrix, LoRA introduces two smaller matrices `A ∈ R^{d × r}` and `B ∈ R^{r × k}`, where `r` (rank) is much smaller than `d` or `k`.

- The original weight update equation is:
  ```
  W_updated = W_0 + ΔW
  ```
  where `ΔW` is the change during fine-tuning.

- With LoRA, the update is approximated using the product of two low-rank matrices:
  ```
  W_updated = W_0 + AB
  ```
  where `A` and `B` are low-rank matrices and `r ≪ min(d, k)`. This makes the update much more efficient because `A` and `B` together have far fewer parameters than `W_0` itself.

#### What is **Rank** in LoRA?

- The rank `r` determines the complexity of the update. If the rank is high, the update matrices `A` and `B` can capture more detailed changes. If the rank is low, the update is simpler and less expressive.
- Mathematically, `A` and `B` represent low-rank factorizations of the full matrix, meaning they approximate the changes in the original matrix without needing all the information.
  
- **Trade-off**: A larger rank `r` allows more complex updates but uses more memory and computation. A smaller rank reduces resource use but limits how much the model can adapt to new data.

---

### 3. **QLoRA (Quantized LoRA)**

QLoRA is the combination of **Quantization** and **LoRA**, which allows for efficient fine-tuning of large language models on low-resource hardware (like consumer-grade GPUs).

Here’s how QLoRA works:

- **Step 1: Quantization**  
  The original model weights are first **quantized** to 4-bit precision, reducing the memory footprint by a factor of 8 (since 32-bit weights are converted to 4-bit). This allows very large models, such as LLaMA-2 7B, to fit on smaller hardware (e.g., GPUs with less VRAM).
  
  Example:
  ```
  W_quantized = Quant(W_0)
  ```
  where `W_0` is the original weight matrix and `W_quantized` is the 4-bit quantized version.

- **Step 2: Low-Rank Adaptation (LoRA)**  
  After quantizing the model, LoRA is applied. The weight matrix is fine-tuned by adding the product of two low-rank matrices:
  ```
  W_fine-tuned = W_quantized + AB
  ```
  where `A` and `B` are the low-rank matrices used to fine-tune the model.

- **Why is this powerful?**  
  By quantizing the large model first and then applying LoRA, the model can be fine-tuned using significantly less memory and computation. This allows even large models like LLaMA-2 to be fine-tuned on consumer hardware, while still achieving strong performance.

### 4. **PEFT (Parameter-Efficient Fine-Tuning)**

**Parameter-Efficient Fine-Tuning (PEFT)** is the general approach behind methods like **LoRA** and **QLoRA**. Instead of updating all the parameters of a large model during fine-tuning, PEFT focuses on updating only a small, efficient subset of parameters. This makes it much more practical to fine-tune large models without needing massive computational resources.

#### Why PEFT?

- Large models, like GPT-3 or LLaMA-2, can have hundreds of billions of parameters. Fine-tuning all of these requires enormous memory and compute power.
- PEFT techniques, like **LoRA**, only adjust a small number of parameters (through low-rank matrices or other efficient methods), meaning that the original model remains mostly unchanged, and only small, targeted updates are made.

Mathematically, PEFT can be described as:
```
W_updated = W_0 + f(ΔW)
```
where `f(ΔW)` represents the updates done efficiently (e.g., using low-rank matrices or adapters). The function `f` could represent LoRA or another method, depending on the PEFT technique used.

---

### Final Summary of Mathematical Concepts

1. **Quantization**:
   - **FP4** and **NF4** reduce the precision of weights from 32/16-bit to 4-bit to save memory.
   - **FP4** uses a fixed precision, while **NF4** adapts dynamically based on data distribution, reducing quantization errors.

2. **LoRA**:
   - Introduces two small matrices, `A` and `B`, to approximate the updates to the original weight matrix `W_0`.
   - **Rank** `r` controls how much the model can adapt. Higher ranks allow for more expressive changes but require more computation.

3. **QLoRA**:
   - Combines quantization with LoRA, where the model is quantized to 4 bits and then fine-tuned using low-rank updates.
   - This allows for highly efficient fine-tuning of large models on hardware with limited resources.

4. **PEFT**:
   - The umbrella concept behind techniques like LoRA and QLoRA. It enables fine-tuning large models by updating only a small subset of parameters, making it practical to fine-tune massive models without needing huge computational resources.
