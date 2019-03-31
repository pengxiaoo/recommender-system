# 排序算法概述
所谓推荐系统中的排序，是指对于召回的候选集中的item，根据其属性，并结合用户画像以及上下文信息，计算出展现优先级的得分，并将items按得分降序排列。从点击率预估的角度，Rank就是估计每个item的点击率，然后按预估点击率对items降序排列。

## 排序模块的常见架构

## 传统排序算法--规则排序

## Learning to Rank

根据模型不同，常见的L2R可以分为LR(Logistic Regression)，gbdt，LR与gbdt的组合，以及深度学习模型等
### LR
### gbdt
### LR+gbdt
### 深度学习模型

根据损失函数定义方式的不同，L2R可以分为三类：Pointwise方法，Pairwise方法，Listwise方法
### Pointwise

### Pairwise

### Listwise
