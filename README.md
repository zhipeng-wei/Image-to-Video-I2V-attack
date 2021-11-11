# Python Environment.
python==3.7.11  
pytorch==1.9.1  
torchvision==0.10.1  
gluoncv==0.11.0  
pandas==1.3.3

# Prepare Model and Dataset
### Video model
For Kinetics-400, download config files from [gluon](https://cv.gluon.ai/model_zoo/action_recognition.html).  Models include i3d_nl5_resnet50_v1_kinetics400, i3d_nl5_resnet101_v1_kinetics400, slowfast_8x8_resnet50_kinetics400, slowfast_8x8_resnet101_kinetics400, tpn_resnet50_f32s2_kinetics400, tpn_resnet101_f32s2_kinetics400.
After that, change the CONFIG_ROOT of utils.py into your custom path. We use pretrained models on Kinetics-400 from gluon to conduct experiments.

We fine-tune these models on UCF-101. Download checkpoint files from [here](https://drive.google.com/drive/folders/10KOlWdi5bsV9001uL4Bn1T48m9hkgsZ2) and specify UCF_CKPT_PATH of utils.py.

### Dataset
Download Kinetics-400 dataset and UCF-101 dataset and set OPT_PATH of utils.py to specify the output path.

For Kinetics-400, change cfg.CONFIG.DATA.VAL_DATA_PATH of utils.py into your validation path.

For UCF-101, split videos into images and change UCF_IMAGE_ROOT of utils.py into your images path of UCF-101.

# Run the code.
### Ablation Study and Performance comparison without white-box video model
```python
python run_image_guided.py --gpu {gpu}
```
The results include Table 2, Figure 4, Figure 5, and Figure 6.

### Generation of adversarial examples
Generate adversarial examples with white-box video models.
For kinetics,
```python
python attack.py --gpu {gpu} --model {model} --attack_type image --attack_method {image_method} --step {step} --batch_size {batch_size} 
python attack.py --gpu {gpu} --model {model} --attack_type video --attack_method TemporalTranslation --step {step} --batch_size 1
```
* model: the white-box model.
* attack_method: such as FGSM, BIM, MI, etc.
* step: the iteration number.

For UCF101,
```python
python attack_ucf101.py --gpu {gpu} --model {model} --attack_type image --attack_method {image_method} --step {step} --batch_size {batch_size} 
python attack_ucf101.py --gpu {gpu} --model {model} --attack_type video --attack_method TemporalTranslation --step {step} --batch_size 1
```
The output path of generated adversarial examples can be directly used to ori_path and adv_path in the next command.

### Performance comparison with white-box video model
Fine-tuning existing adversarial examples by: 
```python
python image_fine_tune_attack.py --gpu {gpu} --attack_method {ILAF or ENS_FT_I2V} --used_ori {ori_path} --used_adv {adv_path} --opt_path {opt_path}
```
* used_ori: the path of original examples.
* used_adv: the path of existing adversarial examples.
* opt_path: the output path.

Predict these generated adversarial examples by 
```python
# ucf101 reference
python reference_ucf101.py --gpu {gpu} --adv_path {adv_path}
# kinetics reference
python reference.py --gpu {gpu} --adv_path {adv_path}
```
* adv_path: the output path of generated adversarial examples.