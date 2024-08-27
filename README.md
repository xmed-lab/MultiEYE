# MultiEYE: Dataset and Benchmark for OCT-Enhanced Retinal Disease Recognition from Fundus Images

This repo is the official implementation of <u>MultiEYE: Dataset and Benchmark for OCT-Enhanced Retinal Disease Recognition from Fundus Images</u>.

<img src="figure\framework.png" style="zoom:43%;" />

To mimic the real clinical circumstance, we formulate a novel setting, "**OCT-enhanced disease recognition from fundus images**", that allows for the use of unpaired multi-modal data during the training phase, and relies solely on the cost-efficient fundus photographs for testing.

To benchmark this setting, we present **the first large multi-modal multi-class dataset for eye disease diagnosis, MultiEYE**, and propose an **OCT-assisted Conceptual Distillation Approach (OCT-CoDA)**, which employs semantically rich concepts to extract disease-related knowledge from OCT images and leverages them into the fundus model.

## Data Preparation

### Dataset

We create a multi-modal multi-disease classification dataset, MultiEYE, by assembling 12 public fundus datasets and 4 OCT datasets with our private data collected from different hospitals. Our dataset is released at [].

### Pre-processing

We use contrast-limited adaptive histogram equalization for fundus images and median filter for OCT images to improve image quality. Also, we adopt data augmentation including random crop, flip, rotation, and changes in contrast, saturation, and brightness. Zero-padding is applied to rectangular images to avoid distortions.

## Implementation

### 1. Environment

Create a new environment and install the requirements:

```shell
conda create -n multieye python==3.10.2
conda activate multieye
pip install -r requirements.txt
```

Check Dependencies:

```
numpy==1.24.4
opencv-python==4.8.1.78
scikit-learn==1.2.2
scipy==1.11.4
torch==1.13.1
torchaudio==0.13.1
torchcam==0.3.2
torchvision==0.14.1
transformers==4.27.4
```

### 2. Training

**Concept Generation**

The first step of our method is to generate a candidate set of concepts describing the specific symptoms of each eye disease. We adopt GPT-4 to autonomously generate attributes for each disease, which are then organized into a list and stored in [`concepts`](concepts).

**OCT Model Pre-training**

We first pretrain the teacher model on OCT images.

```shell
python main_single.py \
--modality "oct" \
--data_path "multieye_data/assemble_oct" \
--concept_path "concepts" \
--batch_size 64 \
--n_classes 9 \
--epochs 100 \
--output_dir "checkpoint/oct_checkpoint" \
--device_id [Select GPU ID]
```

We provide the weights of the pre-trained OCT model at [oct_checkpoint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lwangdk_connect_ust_hk/Eoe0ba8DVtlAsVFohquw2nEB198xIETqtqwkqnci46Yv7g?e=8UgBZI). After downloading, please place the model in the `checkpoint/oct_checkpoint` directory.

**OCT-Assisted Distillation**

Then, we aim to train a fundus classification network assisted by a pre-trained OCT model with the multi-modal data through the proposed OCT-enhanced Conceptual Distillation. 

```shell
python main.py \
--modality "fundus" \
--data_path "multieye_data/assemble" \
--data_path_oct "multieye_data/assemble_oct" \
--concept_path "concepts" \
--checkpoint_path "checkpoint/oct_checkpoint" \ 
--batch_size 64 \
--n_classes 9 \
--epochs 100 \
--alpha_distill 6e-1 \
--beta_distill 5e-2 \
--temperature 10 \
--output_dir "checkpoint/fundus_checkpoint" \
--device_id [Select GPU ID]
```

We provide the weights of our trained model at [fundus_checkpoint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lwangdk_connect_ust_hk/Eoe0ba8DVtlAsVFohquw2nEB198xIETqtqwkqnci46Yv7g?e=8UgBZI).

### 3. Evaluation

Run and evaluate the enhanced fundus model in [`test.ipynb`](test.ipynb).
