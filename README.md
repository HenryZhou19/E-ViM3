## Mamba-3D as Masked Autoencoders for Accurate and Data-Efficient Analysis of Medical Ultrasound Videos
### This repository is built based on [Pytorch Template v1.5](https://github.com/HenryZhou19/Pytorch-Template/tree/v1.5_archive).


## 🚀 Easy Start
### Create a new Python environment and install the requirements
```
conda create -p PATH_TO_ENV python=3.10
conda activate PATH_TO_ENV
# OR
conda create -n ENV_NAME python=3.10
conda activate ENV_NAME
```
```
pip install -r requirements.txt
pip install PATH_TO_CAUSAL_CONV1D_WHL_FILE
pip install PATH_TO_MAMBA_SSM_WHL_FILE
```
For CAUSAL_CONV1D & MAMBA_SSM, please refer to https://github.com/state-spaces/mamba  
We recommend installing `causal_conv1d==1.3.0.post1` and `mamba-ssm==2.2.2`.

### Prepare for datasets (e.g., EchoNet-Dynamic)
#### folder structure:
```
ECHONET-DYNAMIC_DATA_DIR
    ├── Videos
        ├── 0X1A0A263B22CCD966.avi
        ├── 0X1A2A76BDB5B98BED.avi
        ├── ...
        └── 0XFEBEEFF93F6FEB9.avi
    ├── FileList.csv
    └── VolumeTracings.csv
```

### Prepare the config files in the `configs` folder
For example, pre-training E-ViM³-p4 and fine-tuning it for the EF prediction task on EchoNet-Dynamic:
#### Pre-training phase
* Use 'configs/pretrain_echo_mamba_3d_patch4.yaml' as the main config file.
* Set **data.data_dir: YOUR_ECHONET-DYNAMIC_DATA_DIR**
* (Optional) Set **info.output_dir: YOUR_PRETRAIN_OUTPUT_DIR**
* Set other configs as needed.
```
# run pre-training on 2 GPUs without wandb & tensorboard loggers
bash scripts/train.sh -d 0,1 -c pretrain_echo_mamba_3d_patch4 special.no_logger=True
```
#### Fine-tuning phase
* Use 'configs/finetune_echo_mamba_3d_patch4.yaml' as the main config file.
* Set **data.data_dir: YOUR_ECHONET-DYNAMIC_DATA_DIR**
* Set trainer.pretrained_models.pretrained_video_mamba_3d_model: PATH_OF_THE_PTH_FILE
  * PATH_OF_THE_PTH_FILE is the absolute or relative (to the main path of this project) path of the pre-trained model checkpoint file ('.pth') saved in YOUR_PRETRAIN_OUTPUT_DIR/THIS_RUN_FOLDER.
* (Optional) Set **info.output_dir: YOUR_FINETUNE_OUTPUT_DIR**
* Set other configs as needed.
```
# run fine-tuning on 2 GPUs without wandb & tensorboard loggers
bash scripts/train.sh -d 0,1 -c finetune_echo_mamba_3d_patch4 special.no_logger=True
```
#### Inference and evaluation
* Use 'configs/universal_inference.yaml' as the main config file.
```
# run inference on 1 GPU (without wandb & tensorboard loggers as default)
# TRAIN_CFG_PATH is the absolute or relative (to the main path of this project) path of the `cfg.yaml` file in your YOUR_FINETUNE_OUTPUT_DIR/THIS_RUN_FOLDER.
bash scripts/inference.sh -d 0 -p TRAIN_CFG_PATH
```
  
### More instructions on executing bash scripts in the `scripts` folder
* Important notices: 
    * Main entry points are the shell scripts in the 'scripts' folder.  
        All '.sh' files can be executed in Linux CLI by:
        ```
        bash scripts/xxx.sh [OPTIONS] [ADDITIONAL_CONFIGS]
        ```
        * 'bash scripts/xxx.sh -h' will show the short manual of this script.
    * **Tensorboard** (offline logger) and **wandb** (online logger) are enabled by default.  
        So you may pre-register a wandb account to log in when the task is starting.

    * Loggers can be configured in 'yaml' config files, or you can disable all of them by running the bash command with 'special.no_logger=True':
        ```
        bash scripts/train.sh -d 0 -c YAML_FILE_NAME special.no_logger=True
        ```
        * '-c YAML_FILE_NAME' means using './configs/YAML_FILE_NAME.yaml' as the main config file.
    * Most of the configs can be modified or added in the bash command line by adding 'X.Y.Z=abc' intuitively, which equals to the following contents in the 'yaml' file:
        ```
        X:
          Y:
            Z: abc
        ```

* run on CPU, GPU and multiple GPUs by standalone DDP
    ```
    # Example(s):
    # CPU (Not applicable to Mamba-SSM)
    bash scripts/train.sh -d cpu -c YAML_FILE_NAME

    # GPU device_id=0
    bash scripts/train.sh -d 0 -c YAML_FILE_NAME

    # GPU device_id=0,1 (DDP)
    bash scripts/train.sh -d 0,1 -c YAML_FILE_NAME
    ```

* Notes on inference: 
    * It will use 'configs/universal_inference.yaml' as configs for inference by default.

    * The config 'tester.use_best' can control whether to use the best model or the latest model in the outputs folder.

    * Other configs will be the same as in **TRAIN_CFG_PATH** if not specified in the bash command or the inference 'yaml' file.


## Citation
If you find our paper or this repository useful for your research, please consider citing:
```
@misc{zhou2025mamba3dmaskedautoencodersaccurate,
      title={Mamba-3D as Masked Autoencoders for Accurate and Data-Efficient Analysis of Medical Ultrasound Videos}, 
      author={Jiaheng Zhou and Yanfeng Zhou and Wei Fang and Yuxing Tang and Le Lu and Ge Yang},
      year={2025},
      eprint={2503.20258},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.20258}, 
}
```