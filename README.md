# GeoCraft: A Diffusion Model-Based 3D Reconstruction Method Driven by Image and Point Cloud Fusion

## Abstract

With the rapid development of technologies like virtual reality (VR), autonomous driving, and digital twins, the demand for high-precision and realistic multimodal 3D reconstruction has surged. This technology has become a core research focus in computer vision and graphics due to its ability to integrate multi-source data, such as 2D images and point clouds. However, existing methods face challenges such as geometric inconsistency in single-view reconstruction, poor point cloud-to-mesh conversion, and insufficient multimodal feature fusion, limiting their practical application.
To address these issues, this paper proposes GeoCraft, a multimodal 3D reconstruction method that generates high-precision 3D models from 2D images through three collaborative stages: Diff2DPoint, Point2DMesh, and Vision3DGen. Specifically, Diff2DPoint generates an initial point cloud with geometric alignment using a diffusion model and projection feature fusion; Point2DMesh converts the point cloud into a high-quality mesh using an autoregressive decoder-only Transformer and Direct Preference Optimization (DPO); Vision3DGen creates high-fidelity 3D objects through multimodal feature alignment.
Experiments on the Google Scanned Objects (GSO) and Pix3D datasets show that GeoCraft excels in key metrics. On the GSO dataset, its CMMD is 2.810 and FID\textsubscript{CLIP} is 26.420; on Pix3D, CMMD is 3.020 and FID\textsubscript{CLIP} is 27.030. GeoCraft significantly outperforms existing 3D reconstruction methods and also demonstrates advantages in computational efficiency, effectively solving key challenges in 3D reconstruction.


![](Figure/CC11.jpg)



# Project Setup and Requirements

To set up the environment for this project, follow the steps below:


The first step is to create a Conda environment specifically for this project. This helps to ensure that the project dependencies are isolated and do not interfere with other Python projects or system-wide libraries.

You can create a Conda environment named `GeoCraft` by running the following command in your terminal or command prompt:

```bash
conda env create -f environment.yaml
conda activate GeoCraft
```




## Testing

To test and generate predictions from the trained **CCMIM** model, you can use the `test.py` and `inference.py` scripts. The `test.py` script allows you to evaluate the model on a dataset, while `inference.py` lets you test the model on individual images.

### 1. Running Inference on a Test Dataset

To run inference on a test dataset, use the following command:

```bash
python test.py \
--config_dir ./configs \
--checkpoint_dir ./checkpoints/geocraft_train \
--dataset gso \
--save_results \
--visualize \
--eval_metrics CMMD FID_CLIP CLIP-score LPIPS

python test.py \
--config_dir ./configs \
--checkpoint_dir ./checkpoints/geocraft_train \
--dataset pix3d \
--save_results \
--visualize \
--eval_metrics CMMD FID_CLIP CLIP-score LPIPS
```

### 2. Running Inference on a Single Image

To run inference on a single image and generate the visualized detection result, use the following command:
```bash
python inference.py \
  --input_img ../Testing/test_datasets/sample_single_image.jpg \
  --config_dir ./configs \
  --checkpoint_dir ./checkpoints/geocraft_train \
  --output_dir ../Testing/inference_results \
  --num_points 2048 \
  --mesh_post_process \
  --surface_resolution 256 \
  --camera_intrinsics 500 500 256 256 \
  --camera_pose 1 0 0 0 0 1 0 0 0 0 1 2
```

## Training GeoCraft
To train a new GeoCraft model on a dataset, you can use the train.py script. Below is an example of how to run the training process.
```bash
python train.py \
--config_dir ./configs \
--stage all \
--resume False \
--resume_epoch -1
```



## Reuslts

![result](Figure/com11.jpg)

