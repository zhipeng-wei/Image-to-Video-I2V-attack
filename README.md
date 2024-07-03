<div align="center">

<h1><a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Cross-Modal_Transferable_Adversarial_Attacks_From_Images_to_Videos_CVPR_2022_paper.pdf">Cross-Modal Transferable Adversarial Attacks from Images to Videos</a></h1>

**CVPR 2022**

</div>

<div align="center">

<h1><a href="https://ieeexplore.ieee.org/abstract/document/10375740">Adaptive Cross-Modal Transferable Adversarial Attacks From Images to Videos</a></h1>

**IEEE Transactions on Pattern Analysis and Machine Intelligence ( Volume: 46, Issue: 5, May 2024)**

**[Zhipeng Wei](https://zhipeng-wei.github.io/), [Jingjing Chen](https://fvl.fudan.edu.cn/people/jingjingchen), [Zuxuan Wu](https://zxwu.azurewebsites.net/), [Yu-Gang Jiang](https://fvl.fudan.edu.cn/people/yugangjiang/)**
</div>

If you use our method for attacks in your research, please consider citing
```
@inproceedings{wei2022cross,
  title={Cross-Modal Transferable Adversarial Attacks from Images to Videos},
  author={Wei, Zhipeng and Chen, Jingjing and Wu, Zuxuan and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15064--15073},
  year={2022}
}
@ARTICLE{10375740,
  author={Wei, Zhipeng and Chen, Jingjing and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Adaptive Cross-Modal Transferable Adversarial Attacks From Images to Videos}, 
  year={2024},
  volume={46},
  number={5},
  pages={3772-3783},
  keywords={Videos;Adaptation models;Perturbation methods;Feature extraction;Computational modeling;Glass box;Closed box;Cross-modal attack;transferable attack},
  doi={10.1109/TPAMI.2023.3347835}}
```

# Introduction
We proposed the Image To Video (I2V) attack (CVPR paper), which generates adversarial video examples by optimizing against pretrained image models to deceive video models. Specifically, I2V reduces the cosine similarity between adversarial and benign features in the intermediate layers of image models for each video frame.  
Moreover, I2V can be easily extended to simultaneously perturb multi-layer features extracted from an ensemble of image models (TPAMI Paper). To efficiently integrate various features, we introduce an adaptive approach to re-weight the contributions of each layer based on its cosine similarity values of the previous attack step.

# Python Environment.
We provide I2V_attack-env.yml to recover the used environment.
```
conda env create -f I2V_attack-env.yml
```
GPU infos contain 
```
NVIDIA GeForce RTX 2080TI
NVIDIA-SMI 430.14       Driver Version: 430.14       CUDA Version: 10.2 
```

# Prepare Model and Dataset
### Video model
For Kinetics-400, download config files from [gluon](https://cv.gluon.ai/model_zoo/action_recognition.html).  Models include i3d_nl5_resnet50_v1_kinetics400, i3d_nl5_resnet101_v1_kinetics400, slowfast_8x8_resnet50_kinetics400, slowfast_8x8_resnet101_kinetics400, tpn_resnet50_f32s2_kinetics400, tpn_resnet101_f32s2_kinetics400.
After that, change the CONFIG_ROOT of utils.py into your custom path. We use pretrained models on Kinetics-400 from gluon to conduct experiments.

For UCF-101, we fine-tune these models on UCF-101. Download checkpoint files from [here](https://drive.google.com/open?id=10KOlWdi5bsV9001uL4Bn1T48m9hkgsZ2&authuser=weizhipeng1226%40gmail.com&usp=drive_fs) and specify UCF_CKPT_PATH of utils.py.
<!-- (due to the double blind review, we will provide the link after the paper is accepted)  -->

### Dataset
Download Kinetics-400 dataset and UCF-101 dataset and set OPT_PATH of utils.py to specify the output path.

For Kinetics-400, change cfg.CONFIG.DATA.VAL_DATA_PATH of utils.py into your validation path.

For UCF-101, split videos into images and change UCF_IMAGE_ROOT of utils.py into your images path of UCF-101.

# Run the code.
## Ablation Study and Performance Comparison
Using this code to obtain the results of Table 3, Table 4, and Figure 4, Figure 5.
```python
python run_image_guided.py --gpu {gpu}
```
## Generation of adversarial examples
Before comparing the proposed ENS-I2V with ILAF, we need to generate adversarial examples with white-box video models.

For kinetics-400,
```python
python attack.py --gpu {gpu} --model {model} --attack_type image --attack_method {image_method} --step {step} --batch_size {batch_size} 
python attack.py --gpu {gpu} --model {model} --attack_type video --attack_method TemporalTranslation --step {step} --batch_size 1
```
* model: the white-box model.
* attack_method: such as FGSM, BIM, MI, etc. See more attacks in base_attack.py
* step: the iteration number.

For UCF101,
```python
python attack_ucf101.py --gpu {gpu} --model {model} --attack_type image --attack_method {image_method} --step {step} --batch_size {batch_size} 
python attack_ucf101.py --gpu {gpu} --model {model} --attack_type video --attack_method TemporalTranslation --step {step} --batch_size 1
```

These generated adversarial examples will be stored in the OPT_PATH of utils.py, which can be directly used as the parameter of "--used_ori" and "--used_adv" in subsequent commands.

## Comparing against Stronger Baselines
Fine-tuning existing adversarial examples by: 
```python
python image_fine_tune_attack.py --gpu {gpu} --attack_method ILAF --used_ori {path} --used_adv {path} --opt_path {path} --white_model {model} --dataset {dataset}
```
* used_ori: the path of original examples.
* used_adv: the path of existing adversarial examples.
* opt_path: the output path.
* white_model: the white-box model.
* dataset: Kinetics-400 or UCF-101

Predict these generated adversarial examples by 
```python
# ucf101 reference
python reference_ucf101.py --gpu {gpu} --adv_path {adv_path}
# kinetics reference
python reference.py --gpu {gpu} --adv_path {adv_path}
```
* adv_path: the output path of generated adversarial examples.   
