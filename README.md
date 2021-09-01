# hw_text_classification(赛道二)
## **赛题**
### 简介
- 在本次大赛中，主办方提供匿名化的文章质量数据，参赛选手基于给定的数据构建文章质量判别模型。
### 赛题说明
- 题目将为选手提供文章数据，参赛选手用数据构建文章质量判别模型。所提供的数据经过脱敏处理，保证数据安全。
### 评价指标
- F1-score

基础数据集包含两部分：训练集和测试集。其中训练集给定了该样本的文章质量的相关标签；测试集用于计算参赛选手模型的评分指标，参赛选手需要计算出测试集中每个样本文章质量判断及优质文章的类型。
## **解决方案**
### 总体思路
- 本次比赛提供了baseline,所以本代码是在baseline的基础上修改而成的。整体思路为，先用......说实话和baseline 差不多，尝试了一些操作但都没有提升，就不赘述baseline的思路了。
### 数据预处理
- 数据预处理部分采用baseline预处理，去除文章中的空白字符，将所有字符转义为普通字符。
### 模型
- 模型为预训练好的bert-base-chinese 模型，在此基础上接全连接层做finetune。
- 另外还使用了哈工大的xlnet 模型，最后使用不同模型进行融合，得到最终预测结果。
### 基本参数设置
- 详见代码中的train.py中的参数设置。

## **模型训练**
### 设置环境变量
- 推荐使用anaconda 来管理环境变量，根据代码中提供的environment.yaml文件，按照如下步骤复现虚拟环境，确保程序可以正常运行。

```bash
$ conda env create -f environment.yaml
# 执行这一步需要在你的机器中安装anaconda，并且将其添加到环境变量中,确保conda 命令可以正常使用。

```

### 程序执行步骤
```bash
# 切换到code目录下
cd /code
# 数据预处理
$ python preprocess.py
# 训练十分类 bert模型
$ python train.py
# 利用bert模型将所挑选的正负样本生成为PU learning的输入
$ python build_pu_data.py
# 利用生成好的数据训练PU learning 所需的二分类器
$ python train_pu_model.py
# 利用训练好的bert模型和 二分类器对测试集做预测
$ python joint_predictor.py

```

## **系统依赖**

- 操作系统
 ```bash
 LSB Version:    :core-4.1-amd64:core-4.1-noarch:cxx-4.1-amd64:cxx-4.1-noarch:desktop-4.1-amd64:desktop-4.1-noarch:languages-4.1-amd64:languages-4.1-noarch:printing-4.1-amd64:printing-4.1-noarch
 Distributor ID: CentOS
 Description:    CentOS Linux release 7.8.2003 (Core)
 Release:        7.8.2003
 Codename:       Core
 ```
 - python 3.6
 - cuda 10.2
