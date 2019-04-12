# 学习排序简介
学习排序(Learning to Rank)是指采用机器学习的方式来解决排序问题(Ranking Problem)。以信息检索系统为例，一个信息检索系统存储了大量的documents，每当用户提交一个query，系统从数据库中获取包含关键词的documents，然后对这些documents按照与query的相关程度排序，并返回相关程度最高的N个documents给用户。

document和query的相关度可以定义为f(q,d)，q代表一个query，d代表一个document，f(q,d)取实数值，当q和d相关程度越高时f值越大。f(q,d)通常被称为排序模型。这样一个排序问题就可定义为给定query和documents，如何找到最优的排序模型f(q,d)。而学习排序就是用有监督学习的方式，自动找到最优的排序模型。

由于学习排序最初起源于信息检索领域，因此学习排序相关的文献通常会沿用信息检索系统中的术语，例如query和document，本文也将如此。query和document分别对应于推荐系统术语中的user和item。

## 测试集和训练集
从严格定义的角度，学习排序的每一条training data形如(q, D=[d1,d2,..,dn], L=[l1,l2,..,ln])。其中q为query，D为与该query相关的documents列表，假设长度为n（n的大小与q有关，这里为了符号的简便省略了q）。L是标签的列表，L和D的长度相同并且两者的元素一一对应。每一个标签l代表相应document和query的相关程度的级别，它的取值范围为枚举：可以是二元（相关、不相关），也可以是多元（完全无关、部分相关、强相关等）。

在实践中，由于人工打标签成本太高，所以很多时候会用一些统计量（例如点击次数，点击次序）来代替。例如一条training data可以是(q, D=[d1,d2,..,dn], C=[c1,c2,..,cn])，其中ci代表di被查询者点击的次数。

学习排序的每一条testing data形如(q,T=[t1,t2,..,tm])，其中q为query，T为候选documents的集合。如前所述，学习排序的训练目标是获得最优的排序模型f(q,d)。当训练完成并进入inference阶段时，排序模型会对每一对候选的(q,d=ti)进行打分，从而得到相关度从高到低的排序。

可以看出，学习排序属于Structured Learning。它的输入是一个有序序列(Sequence)，输出是对于这个序列的评分。

## 排序模型的评估
理想情况下，排序模型的评估标准需要满足几个要求：
>* 倾向于选择能把相关度高的document排在前面的排序模型。
>* 当列表中documents顺序不对的时候，也能对列表的排序做出合理的打分。例如已知ground truth是，某个query对应的top3 documents分别是d1,d2,d3，那么对于d1,d3,d2和d3,d2,d1这两个排序，能判断出前者优于后者。
>* 要考虑到不同位序的影响权重的不同。例如，假如最终排序结果是长度为100的document列表，那么前10个documents当中出现错序要比第91到第100个document当中出现错序更加严重。
>* 要便于计算。

排序学习中常用的评估准则有AUC，DCG，nDCG，MAP(Mean Average Precision)等等
### AUC(Area Under ROC Curve)
AUC是点击率预估问题中常用的评估准则，对于排序学习而言也有一定的参考价值。直观上理解，AUC代表任意取一对正负样本(q,d+)和(q,d-)，正样本相关度得分大于负样本的概率（这里正样本d+代表对于给定的query，用户确实点击了该document；而负样本d-代表该document展示给了用户但没有获得用户点击）。AUC的计算方法如下：
>* 遍历正负样本，将正样本和负样本两两组对[pair1,pair2,...,pairP]，P代表总的pair数；
>* 遍历每一个pair，如果该pair中正样本的得分大于负样本的得分，那么认为该pair被正确判断；
>* 正确判断的pair个数除以总的pair个数P，就是AUC。
可以看出，AUC的取值范围介于0到1之间。对于完美排序，AUC取值为1；如果是顺序是完全随机的，AUC取值为0.5左右。AUC的缺点是没有考虑到位序对评分权重的影响。

### DCG(Discounted Cumulative Gain)
DCG假定一个排序结果的总评分由该排序结果中每个document的评分累加得到；而每个document的评分由该document与query的相关度、以及该document在排序结果中的位置决定。DCG的公式如下
![Image text](https://github.com/pengxiaoo/recommender-system/blob/master/imgs/DCG.png)
上式中，p代表DCG是针对排序结果前p个documents计算得到的；reli代表第i个document与query的相关度；分母的log的意义是对每一个document的权重按照位置进行对数衰减。可以看出，DCG既考虑到了document与query的相关度，又考虑到了document的位置因素，因而是排序学习当中比较主流的评估准则。
### nDCG(normalized DCG)
在实际应用中，不同query对应的documents列表长度往往差别较大，因此DCG指标需要进行归一化。常用的归一化方法是用原始DCG除以IDCG，其中IDCG是指query对应的前p条documents的完美排序列表的DCG值。原始DCG除以IDCG的结果被称为nDCG(normalized DCG)。
![Image text](https://github.com/pengxiaoo/recommender-system/blob/master/imgs/IDCG.png)
可以看出，nDCG的取值范围在0到1之间。

## 学习排序的分类
理想情况下，学习排序的loss function可定义为1 - nDCG。当nDCG=1的时候为完美排序，此时获得zero loss。不过在实践中，loss function有时会采用简化的定义方式。根据损失函数定义方式的不同，学习排序可以分为三类：Pointwise方法，Pairwise方法，Listwise方法。

在介绍三种分类之前，我们先调整一下术语和符号，以便更符合机器学习的传统。我们把排序模型f(q,d)写成f(x)，其中x代表由q和d联合组成的特征；同时把label写成y。

### Pointwise方法
pointwise方法就是独立计算每一个document的相关度得分，然后将不同documents按得分高低排序。pointwise方法的损失函数定义如下

![Image text](https://github.com/pengxiaoo/recommender-system/blob/master/imgs/pointwise-loss.png)

对于一个特定的query，每一个document独立计算loss，然后将所有documents的loss累加起来。pointwise loss相当于1 - nDCG的上界。

从实践的角度，pointwise方法既可以建模成回归问题（如果预测目标为连续值），也可以建模成分类问题（如果预测目标为离散值，例如相关度级别）。

pointwise方法比较简单易行，在实际中应用的也较为广泛。依据采用的模型不同，可以分为
>* 单一浅层模型。例如LR，SVM等。特点是速度快，支持特征的维度高，推荐结果可解释性较好，但是需要人工特征筛选、特征的归一化和离散化、特征组合，对工程师的特征工程能力有较高要求。另外浅层模型难以捕获交叉特征的影响。
>* 树模型：例如gbdt。树模型不需要特征归一化和特征筛选，有比较强的特征筛选能力。
>* gbdt和LR的组合。这个方法源自一篇著名的文章[Practical Lessons from Predicting Clicks on Ads at Facebook][3]，其中gbdt用来抽取特征，LR基于gbdt输出的特征进行分类，从而省去了人工寻找特征、特征组合的步骤。gbdt和LR的组合在多项评估指标上明显好过单一模型。但是由于经历了两层模型，推荐结果可解释性较差。
>* 神经网络模型：如今在业界主排序算法中，基于神经网络的深度学习模型得到了越来越多的采用。例如[美团“猜你喜欢”深度学习排序模型实践][4]这篇文章提到了基于Multi-task DNN的深度学习排序。神经网络模型的表达能力和通用性较强，但是也有推荐结果可解释性差的缺点。

### Pairwise方法
在pointwise方法中，排序完全基于单个document的独立得分，没有考虑documents之间的顺序。pairwise方法重点判断任意两个document组成的pair即<document1,document2>是否满足顺序关系。pairwise方法的损失函数定义如下

![Image text](https://github.com/pengxiaoo/recommender-system/blob/master/imgs/pairwise-loss.png)
其中sign(yi-yj)代表documenti和documentj的顺序先后关系，f(xi)-f(xj)代表documenti和documentj的得分高低关系。
Φ代表了如果documenti和documentj的顺序先后关系与得分高低关系不兼容，那么需要付出多大的损失。类似pointwise loss，pairwise loss也是1 - nDCG的上界。
根据Φ的定义方式不同，pairwise方法有不同的实现：
>* Ranking SVM，Φ定义为hinge loss
>* RankingBoost，Φ定义为exponential loss
>* RankingNet，Φ定义为logistic loss

### Listwise方法
pointwise方法对每个document独立算分，pairwise方法对每一对document pair算分，而listwise方法则是针对一个完整document列表算分。不同的listwise方法当中，损失函数的定义稍有不同。例如AdaRank当中，listwise方法的损失函数定义如下

![Image text](https://github.com/pengxiaoo/recommender-system/blob/master/imgs/listwise-loss.png)

与pointwise和pairwise把原始问题拆分简化求解不同，listwise把学习排序作为Structured Learning问题直接求解，充分利用了每一条training data当中documents列表的结构信息，损失函数的定义也跟nDCG直接相关。常用的listwise方法有AdaRank，ListNet，SVM MAP等等。


[1]: http://times.cs.uiuc.edu/course/598f14/l2r.pdf
[2]: https://tech.meituan.com/2018/12/20/head-in-l2r.html
[3]: https://quinonero.net/Publications/predicting-clicks-facebook.pdf
[4]: https://tech.meituan.com/2018/03/29/recommend-dnn.html
