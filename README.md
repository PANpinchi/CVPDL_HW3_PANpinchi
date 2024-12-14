# CVPDL HW3: Object Detection 

## Recap: Object Detection 
### Object Detection 
* Input: 2D RGB image 
* Task: localization and classification
* Output: N x [points, confidence]

### Data imbalance
* There is an imbalance in the distribution of data across categories. Addressing this imbalance is crucial for developing effective and unbiased models.

## Background Information: Foundation Models
### What are Foundation Models?
* Large-scale, pre-trained models having been developed using vast amounts of data can be adapted to accomplish a broad range of tasks.
* Examples:
    * BERT (Question Answering, Translation)
    * GPT (ChatGPT)
    * Claude (Programming)
    * Stable Diffusion (T2I Generation)
    * BLIP2 (Visual Question Answering)
    * …

## Background Information: Layout-to-Image methods
The current Stable Diffusion model relies on class labels or text prompts for conditioning. To enable precise control, such as positioning objects in a layout, methods like fine-tuning and cross-attention map adjustments have been developed.

## Goals of HW3
* We want to leverage two Foundation Models, BLIP2 and Stable Diffusion, to solve the imbalance problem of HW1_dataset.
* Considering that Stable-diffusion-based methods require text prompts as inputs for generation, we can first generate prompts from the given dataset by the image captioning ability of BLIP2.
* After obtaining text prompts for later image generation, one problem still needs to be solved. That is, object detection demands bounding boxes for training.
* Thus, we utilize layout-to-image method to guide the Stable Diffusion model in generating objects in the regions defined by the bounding boxes from label.json.


## Getting Started 
```bash
# Clone the repo:
git clone https://github.com/PANpinchi/CVPDL_HW3_PANpinchi.git
# Move into the root directory:
cd CVPDL_HW3_PANpinchi
```

## Environment Settings
```bash
# Create a virtual conda environment:
conda create -n cvpdl_hw1 python=3.10

# Activate the environment:
conda activate cvpdl_hw1

# Install PyTorch, TorchVision, and Torchaudio with CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Install additional dependencies from requirements.txt:
pip install -r requirements.txt
```
## Download the Required Data
#### 1. Pre-trained Models
Run the commands below to download the pre-trained DETR model. 
```bash
cd detr

# The pre-trained model. 
gdown --folder https://drive.google.com/drive/folders/1XcTS6IyxczQuX703tUTLH8ECMUQGw8Ew?usp=drive_link

cd ..
```
Note: `checkpoint.pth` files should be placed in the `./detr/outputs` folders.

#### 2. Datasets
Run the commands below to download the HW1 datasets.
```bash
gdown --id 1lWdAdjRvQHULW2AbDeZiR-S4Qw5SQ3nE

unzip cvpdl_hw1.zip
```

#### The data directory structure should follow the below hierarchy.
```
${ROOT}
|-- test
|   |-- images
|   |   |-- xxx-xxx-xxx.jpeg
|   |   |-- xxx-xxx-xxx.jpeg
|   |   |-- ...
|   |   |-- xxx-xxx-xxx.jpeg
|-- train
|   |-- images
|   |   |-- xxx-xxx-xxx.jpeg
|   |   |-- xxx-xxx-xxx.jpeg
|   |   |-- ...
|   |   |-- xxx-xxx-xxx.jpeg
|   |-- labels
|   |   |-- xxx-xxx-xxx.txt
|   |   |-- xxx-xxx-xxx.txt
|   |   |-- ...
|   |   |-- xxx-xxx-xxx.txt
|-- valid
|   |-- images
|   |   |-- xxx-xxx-xxx.jpeg
|   |   |-- xxx-xxx-xxx.jpeg
|   |   |-- ...
|   |   |-- xxx-xxx-xxx.jpeg
|   |-- labels
|   |   |-- xxx-xxx-xxx.txt
|   |   |-- xxx-xxx-xxx.txt
|   |   |-- ...
|   |   |-- xxx-xxx-xxx.txt
```

## 【Dataset Preprocessing】
#### Run the commands below to preprocess the dataset.
```bash
python convert2coco.py
```

## 【Training】
#### Run the commands below to train the DETR model.
```bash
python detr/main.py --batch_size=32 --num_workers=2 --epochs=50 \
  --num_classes=17 --num_queries=100 --data_path='.' \
  --train_folder='./train/images' --train_json='./train.json' \
  --val_folder='./valid/images' --val_json='./valid.json' \
  --output_dir='./detr/outputs' \
  --resume='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
```

## 【Testing】
#### Run the commands below to test the DETR model.
```bash
python test.py
```

## 【Evaluation】
#### Run the commands below to evaluate the results. (mAP only
```bash
python eval.py ./tmp_results/predictions_10.json valid_target.json 

python eval.py ./predictions/valid_r12942103.json valid_target.json 

python eval.py ./predictions/test_r12942103.json <test_target.json>
```

#### Run the commands below to evaluate the results. (mAP, AP50, AP75
```bash
python eval_ours.py ./tmp_results/predictions_10.json valid_target.json 

python eval_ours.py ./predictions/valid_r12942103.json valid_target.json 

python eval_ours.py ./predictions/test_r12942103.json <test_target.json>
```

