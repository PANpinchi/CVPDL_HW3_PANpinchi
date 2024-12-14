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
conda create -n cvpdl_hw3 python=3.10

# Activate the environment:
conda activate cvpdl_hw3

# Install PyTorch, TorchVision, and Torchaudio with CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Install additional dependencies from requirements.txt:
pip install -r requirements.txt
```
## Download the Required Data
#### 1. Clone GitHub Repositories
Run the commands below to clone GLIGEN and Stable Diffusion repositories.
```bash
git clone https://github.com/gligen/GLIGEN.git

git clone https://github.com/CompVis/stable-diffusion.git
```
```bash
# move file to repositories
mv gligen_inference_cvpdl_hw3.py GLIGEN
mv txt2img_cvpdl_hw3.py stable-diffusion/scripts
```

#### 2. Datasets
Run the commands below to download the HW3 datasets.
```bash
gdown --id 1t6bFlf-hdQwiyJPTvcKblbPlR3qL_q4_

unzip cvpdl_hw3.zip
```
#### 3. Pre-trained Models
Run the commands below to download the pre-trained GLIGEN model. 
```bash
cd GLIGEN

mkdir gligen_checkpoints

cd gligen_checkpoints

wget -O checkpoint_generation_text.pth https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin

wget -O checkpoint_generation_text_image.pth https://huggingface.co/gligen/gligen-generation-text-image-box/resolve/main/diffusion_pytorch_model.bin

cd ../..
```
Run the commands below to download the pre-trained Stable Diffusion model. 
```bash
cd stable-diffusion/models/ldm

mkdir stable-diffusion-v1

cd stable-diffusion-v1

wget -O model.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt

cd ../../../..
```

## 【Stage 1: Image Captioning and Prompt Design】
#### Run the commands below to install transformers.
```bash
pip install git+https://github.com/huggingface/transformers.git
```
#### Run the commands below to perform image captioning.
```bash
python image_captioning.py
```

## 【Stage 2: Text-to-Image Generation for Data Augmentation】
#### Run the commands below to perform text-to-image generation. (text only)
```bash
# install transformers
pip install git+https://github.com/huggingface/transformers.git

cd stable-diffusion

python scripts/txt2img_cvpdl_hw3.py \
    --label_file <LABEL_FILE> \
    --prompt_type <PROMPT_TYPE> \
    --outdir <OUTDIR>

# Example
python scripts/txt2img_cvpdl_hw3.py \
    --label_file "../visualiztion_200_with_blip2-opt-6.7b-coco.json" \
    --prompt_type "prompt_w_label" \
    --outdir "outputs/text2imgs_with_blip2-opt-6.7b-coco_prompt_w_label"
```
### Available Label File
The following json file can be specified with the `--label_file` option:
- `../visualiztion_200_with_blip2-opt-6.7b-coco.json`
- `../label_with_blip2-opt-2.7b.json`
- `../label_with_blip2-opt-6.7b-coco.json`
- `../label_with_blip2-opt-6.7b.json`
- `../label_with_blip2-flan-t5-xl.json`

### Available Prompt Type
The following prompt type can be specified with the `--prompt_type` option:
- `generated_text`
- `prompt_w_label`
- `prompt_w_suffix`


#### Run the commands below to perform layout-to-image generation.
#### (text + layout / text + layout + reference images)
```bash
# install transformers
pip install transformers==4.19.2

cd GLIGEN

python gligen_inference_cvpdl_hw3.py \
    --label_file <LABEL_FILE> \
    --model_type <MODEL_TYPE> \
    --prompt_type <PROMPT_TYPE> \
    --outdir <OUTDIR>

# Example
python gligen_inference_cvpdl_hw3.py \
    --label_file "../visualiztion_200_with_blip2-opt-6.7b-coco.json" \
    --model_type "box_text_image" \
    --prompt_type "prompt_w_label" \
    --outdir "layout2imgs_with_blip2-opt-6.7b-coco_prompt_w_label"
```

### Available Label File
The following json file can be specified with the `--label_file` option:
- `../visualiztion_200_with_blip2-opt-6.7b-coco.json`
- `../label_with_blip2-opt-2.7b.json`
- `../label_with_blip2-opt-6.7b-coco.json`
- `../label_with_blip2-opt-6.7b.json`
- `../label_with_blip2-flan-t5-xl.json`

### Available Model Type
The following prompt type can be specified with the `--model_type` option:
- `box_text`
- `box_text_image`

### Available Prompt Type
The following prompt type can be specified with the `--prompt_type` option:
- `generated_text`
- `prompt_w_label`
- `prompt_w_suffix`



## 【Evaluation】
#### Run the commands below to evaluate the results. (FID only
```bash
pip install pytorch_fid
```
```bash
python -m pytorch_fid <Path_to_dataset1> <Path_to_dataset2>
```

