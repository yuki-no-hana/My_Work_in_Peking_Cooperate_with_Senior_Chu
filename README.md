# Extra Data
[COCOText](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=5&f=aHR0cDovL2RhdGFzZXRzLmN2Yy51YWIuZXMvcnJjL0NPQ08tVGV4dC13b3Jkcy10cmFpbnZhbC56aXA=)

[SynthTextAdd](https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition) (ref:[SAR](https://arxiv.org/abs/1811.00751))

# Prepare Data
```
ln -s /data1/chuxiaojie/Datasets/data_lmdb_release/ ./
```
## Training Set Directory of data_lmdb_release 
```
# MJSynth 
data_lmdb_release/training/MJ   

# training set of COCOText
data_lmdb_release/training/real_COCO  

# training set of IC13,IC15,IIIT,SVT
# same with data_lmdb_release/validation
data_lmdb_release/training/real_valid  

# SynthText (w/o special characters)
data_lmdb_release/training/ST   

# SynthText (w/ special characters)
data_lmdb_release/training/ST_spe 

# SynthTextAdd
data_lmdb_release/training/SynthTextAdd
```

# Example
默认输入图像大小为32x100，（TPSHD的输出为32x100，在大图64x256上进行采样）。

图像为rgb图像，数据集为MJ、ST、STAdd以及真实数据集（IC13,IC15,IIIT,SVT和COCOText的训练集）。

训练策略为lr=0.001,使用cosine lr_scheduler，前6000步（30k的0.02)作线性warmup。

默认会每2000步在测试集上进行测试，保存总精度最高的模型。

## TPS-VGG-BiLSTM-CTC
训练
```
CUDA_VISIBLE_DEVICES=0 python3 run_train.py --Transformation=TPS --FeatureExtraction=ResNet --SequenceModeling=BiLSTM --Prediction=CTC --select_data=MJ-S-real --batch_ratio=0.484375-0.484375-0.03125  --imgH=32 --imgW=100 --output_channel=512 --hidden_size=512 --seed=0 --lr=0.001 --scheduler=Cosine --warmup_a=0.02 --rgb --train
```
测试（把最后的--train 改成 --test即可）
```
CUDA_VISIBLE_DEVICES=0 python3 run_train.py --Transformation=TPS --FeatureExtraction=ResNet --SequenceModeling=BiLSTM --Prediction=CTC --select_data=MJ-S-real --batch_ratio=0.484375-0.484375-0.03125  --imgH=32 --imgW=100 --output_channel=512 --hidden_size=512 --seed=0 --lr=0.001 --scheduler=Cosine --warmup_a=0.02 --rgb --test
```
## TPSHD-VGG-BiLSTM-CTC
```
CUDA_VISIBLE_DEVICES=0 python3 run_train.py --Transformation=TPSHD --FeatureExtraction=VGG --SequenceModeling=BiLSTM --Prediction=CTC --select_data=MJ-S-real --batch_ratio=0.484375-0.484375-0.03125  --imgH=64 --imgW=256 --output_channel=512 --hidden_size=512 --seed=0 --lr=0.001 --scheduler=Cosine --warmup_a=0.02 --rgb --train 
```
## TPSHD-ResNet-BiLSTM-CTC
```
CUDA_VISIBLE_DEVICES=0 python3 run_train.py --Transformation=TPSHD --FeatureExtraction=ResNet --SequenceModeling=BiLSTM --Prediction=CTC --select_data=MJ-S-real --batch_ratio=0.484375-0.484375-0.03125  --imgH=64 --imgW=256 --output_channel=512 --hidden_size=512 --seed=0 --lr=0.001 --scheduler=Cosine --warmup_a=0.02 --rgb --train 
```
## TPS-ResNet-BiLSTM-AttnGRU
```
CUDA_VISIBLE_DEVICES=0 python3 run_train.py --Transformation=TPS --FeatureExtraction=ResNet --SequenceModeling=BiLSTM --Prediction=AttnGRU --select_data=MJ-S-real --batch_ratio=0.484375-0.484375-0.03125  --imgH=32 --imgW=100 --output_channel=512 --hidden_size=512 --seed=0 --lr=0.001 --scheduler=Cosine --warmup_a=0.02 --rgb --train 
```
## TPSHD-ResNet-BiLSTM-CTC(finetune)
```
 CUDA_VISIBLE_DEVICES=0 python3 run_train.py  --Transformation TPSHD --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC --imgH=64 --imgW=256 --select_data=MJ-S-real --batch_ratio=0.484375-0.484375-0.03125 --lr=0.001 --scheduler=Cosine --warmup_a=0.02 --hidden_size=512 --num_iter=300000 --seed=0 --rgb --saved_model=/data/chuxiaojie/projects/STR/deep-text-recognition-benchmark/result/TPSHD-ResNet-BiLSTM-AttnGRU-MJ-S-real_0.484375-0.484375-0.03125_512-0.001Cosine0.02-Seed0_best_accuracy.pth/best_accuracy.pth --annotation=ft --FT --train 
```

## TPSHD-ResNetfamal-BiLSTM-AttnGRU
```
CUDA_VISIBLE_DEVICES=9 python3 run_train.py  --Transformation TPSHD --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC --imgH=64 --imgW=256 --select_data=MJ-S-real --batch_ratio=0.484375-0.484375-0.03125 --lr=0.001 --scheduler=Cosine --warmup_a=0.02 --hidden_size=512 --train --num_iter=300000 --seed=0 --rgb --saved_model=/data/chuxiaojie/projects/STR/deep-text-recognition-benchmark/result/TPSHD-ResNet-BiLSTM-AttnGRU-MJ-S-real_0.484375-0.484375-0.03125_512-0.001Cosine0.02-Seed0_best_accuracy.pth/best_accuracy.pth --annotation=ft --FT
```

# Modules
## TPSHD
结构与TPS一致，但Localization Network在32x64的大小上进行计算，并从64x256(输入图像大小)的图像中采样出32x100的图。
## AttnGRU
原论文(Attn)使用LSTM，而AttnGRU使用GRU。
# Result
## TPSHD-ResNet-BiLSTM-CTC
accuracy: IIIT5k_3000: 94.967   SVT: 92.427     IC03_860: 95.465        IC03_867: 94.810        IC13_857: 95.216        IC13_1015: 94.286       IC15_1811: 83.987       IC15_2077: 80.116       SVTP: 82.481    CUTE80: 88.542  total_accuracy: 89.790        averaged_infer_time: 0.687      # parameters: 55.432

accuracy: IIIT5k_3000: 95.167   SVT: 90.881     IC03_860: 94.767        IC03_867: 95.040        IC13_857: 95.566        IC13_1015: 94.680       IC15_1811: 84.539   IC15_2077: 80.453       SVTP: 81.860    CUTE80: 86.458  total_accuracy: 89.840  averaged_infer_time: 0.983      # parameters: 55.432

accuracy: IIIT5k_3000: 94.867   SVT: 91.808     IC03_860: 95.349        IC03_867: 94.579        IC13_857: 95.333        IC13_1015: 93.990       IC15_1811: 83.876  IC15_2077: 79.730       SVTP: 84.496    CUTE80: 87.847  total_accuracy: 89.699  averaged_infer_time: 0.673      # parameters: 55.432

## TPS-ResNet-BiLSTM-CTC
accuracy: IIIT5k_3000: 95.067   SVT: 90.572     IC03_860: 94.651        IC03_867: 94.810        IC13_857: 94.982        IC13_1015: 93.793       IC15_1811: 83.932  IC15_2077: 79.779       SVTP: 85.271    CUTE80: 87.500  total_accuracy: 89.658  averaged_infer_time: 0.667      # parameters: 55.432

accuracy: IIIT5k_3000: 95.000   SVT: 91.345     IC03_860: 95.116        IC03_867: 94.464        IC13_857: 94.982        IC13_1015: 93.892       IC15_1811: 84.263  IC15_2077: 80.260       SVTP: 82.481    CUTE80: 85.764  total_accuracy: 89.641  averaged_infer_time: 0.657      # parameters: 55.432

## TPSHD-VGG-BiLSTM-CTC
accuracy: IIIT5k_3000: 93.067   SVT: 89.335     IC03_860: 94.419        IC03_867: 93.887        IC13_857: 94.632        IC13_1015: 93.596       IC15_1811: 81.944  IC15_2077: 77.853       SVTP: 81.550    CUTE80: 80.556  total_accuracy: 87.975  averaged_infer_time: 0.491      # parameters: 16.718

accuracy: IIIT5k_3000: 93.933   SVT: 89.335     IC03_860: 93.953        IC03_867: 93.887        IC13_857: 93.232        IC13_1015: 92.118       IC15_1811: 81.778  IC15_2077: 77.997       SVTP: 83.101    CUTE80: 83.333  total_accuracy: 88.083  averaged_infer_time: 0.491      # parameters: 16.718

## None-VGG-BiLSTM-CTC
accuracy: IIIT5k_3000: 91.433   SVT: 88.253     IC03_860: 93.953        IC03_867: 93.772        IC13_857: 93.466        IC13_1015: 91.921       IC15_1811: 78.410  IC15_2077: 74.627       SVTP: 78.605    CUTE80: 78.819  total_accuracy: 85.962  averaged_infer_time: 0.321      # parameters: 15.025

accuracy: IIIT5k_3000: 91.233   SVT: 88.099     IC03_860: 93.721        IC03_867: 93.080        IC13_857: 93.116        IC13_1015: 91.921       IC15_1811: 77.692  IC15_2077: 74.338       SVTP: 78.450    CUTE80: 78.125  total_accuracy: 85.630  averaged_infer_time: 0.328      # parameters: 15.025

## TPS-ResNet-BiLSTM-AttnGRU
accuracy: IIIT5k_3000: 96.167   SVT: 92.890     IC03_860: 95.930        IC03_867: 96.194        IC13_857: 96.266        IC13_1015: 94.483       IC15_1811: 86.140  IC15_2077: 82.667       SVTP: 87.132    CUTE80: 92.361  total_accuracy: 91.439  averaged_infer_time: 3.083      # parameters: 57.592

accuracy: IIIT5k_3000: 96.100   SVT: 93.663     IC03_860: 96.047        IC03_867: 96.078        IC13_857: 96.499        IC13_1015: 95.369       IC15_1811: 86.251  IC15_2077: 82.523       SVTP: 88.682    CUTE80: 94.444  total_accuracy: 91.680  averaged_infer_time: 3.056      # parameters: 57.592

## TPSHD-ResNet-BiLSTM-AttnGRU
accuracy: IIIT5k_3000: 96.033   SVT: 93.663     IC03_860: 96.395        IC03_867: 96.078        IC13_857: 97.200        IC13_1015: 95.961       IC15_1811: 87.024  IC15_2077: 83.486       SVTP: 88.527    CUTE80: 90.972  total_accuracy: 91.978  averaged_infer_time: 2.997      # parameters: 57.592

accuracy: IIIT5k_3000: 96.267   SVT: 94.745     IC03_860: 96.628        IC03_867: 96.540        IC13_857: 97.200        IC13_1015: 95.172       IC15_1811: 86.803  IC15_2077: 83.293       SVTP: 87.907    CUTE80: 90.972  total_accuracy: 91.978  averaged_infer_time: 2.925      # parameters: 57.592

## TPSHD-ResNetfamal-BiLSTM-AttnGRU
accuracy: IIIT5k_3000: 96.700   SVT: 94.436     IC03_860: 96.628        IC03_867: 96.424        IC13_857: 97.900        IC13_1015: 96.355       IC15_1811: 87.521       IC15_2077: 85.508       SVTP: 88.992    CUTE80: 92.014  total_accuracy: 92.782        averaged_infer_time: 1.696      # parameters: 58.084

accuracy: IIIT5k_3000: 96.767   SVT: 93.354     IC03_860: 96.860        IC03_867: 96.655        IC13_857: 97.550        IC13_1015: 95.862      IC15_1811: 88.128       IC15_2077: 85.749       SVTP: 89.922    CUTE80: 90.972  total_accuracy: 92.865  averaged_infer_time: 1.699     # parameters: 58.084

## TPSHD-ResNet-BiLSTM-CTC(finetune)
accuracy: IIIT5k_3000: 95.767   SVT: 93.509     IC03_860: 95.698        IC03_867: 95.732        IC13_857: 95.799        IC13_1015: 94.680      IC15_1811: 86.306       IC15_2077: 82.860       SVTP: 86.357    CUTE80: 90.278  total_accuracy: 91.274  averaged_infer_time: 0.921     # parameters: 55.432

accuracy: IIIT5k_3000: 95.667   SVT: 92.272     IC03_860: 95.465        IC03_867: 95.271        IC13_857: 96.383        IC13_1015: 95.764      IC15_1811: 85.699       IC15_2077: 81.897       SVTP: 85.116    CUTE80: 89.236  total_accuracy: 90.917  averaged_infer_time: 0.926     # parameters: 55.432