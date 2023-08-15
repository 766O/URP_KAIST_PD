# URP_KAIST_PD

SSD 기반 Pedestraian Detection
  

[SSD paper](https://github.com/](https://arxiv.org/abs/1512.02325)https://arxiv.org/abs/1512.02325)

[SSD git hub](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)

#

### 1.Base line model

| Model | MR(all) | MR(day) | MR(night) | Recall |
| :-----: | :---: | :---: | :---: | :---: |
| Baseline_RGB | 34.78 | 32.15 | 40.94 | 77.23 |
| Baseline_Thermal | 31.38 | 37.76 | 17.83 | 81.70 |


#

### 2.Halfway fusion model
| Model | MR(all) | MR(day) | MR(night) | Recall |
| :-----: | :---: | :---: | :---: | :---: |
| Halfwayfusion+bn | 21.85 | 20.66 | 24.29 | 86.45 |

![image](https://github.com/766O/URP_KAIST_PD/assets/121467486/13e9557b-f78b-40fb-92bc-30d2235f9364)



# Example Image

- Day
![image](https://github.com/766O/URP_KAIST_PD/assets/121467486/680834e9-add3-4152-b1d1-411f7f6e9fcf)

- Night
![image](https://github.com/766O/URP_KAIST_PD/assets/121467486/b9ac9513-79f3-4f1d-930a-8bd67c512ba7)


