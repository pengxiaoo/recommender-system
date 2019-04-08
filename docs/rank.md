# 学习排序简介
学习排序(Learning to Rank)是指采用机器学习的方式来解决排序问题(Ranking Problem)。以信息检索系统为例，一个信息检索系统存储了大量的documents，每当用户提交一个query，系统从数据库中获取包含关键词的documents，然后对这些documents按照与query的相关程度排序，并返回相关程度最高的N个documents给用户。

document和query的相关度可以定义为f(q,d)，q代表一个query，d代表一个document，f(q,d)取实数值，当q和d相关程度越高时f值越大。f(q,d)通常被称为排序模型。这样一个排序问题就可定义为给定query和documents，如何找到最优的排序模型f(q,d)。而学习排序就是用有监督学习的方式，自动找到最优的排序模型。

关于学习排序的formulation，可以参考文章[A Short Introduction to Learning to Rank][1]。

由于学习排序最初起源于信息检索领域，因此学习排序相关的文献通常会沿用信息检索系统中的术语，例如query和document，本文也将如此。query和document分别对应于推荐系统术语中的user和item。

## 测试集和训练集
从严格定义的角度，学习排序的每一条training data形如(q, D=[d1,d2,..,dn], L=[l1,l2,..,ln])。其中q为query，D为与该query相关的documents列表，假设长度为n（n的大小与q有关，这里为了符号的简便省略了q）。L是标签的列表，L和D的长度相同并且两者的元素一一对应。每一个标签l代表相应document和query的相关程度的级别，它的取值范围为枚举：可以是二元（相关、不相关），也可以是多元（完全无关、部分相关、强相关等）。

在实践中，由于人工打标签成本太高，所以很多时候会用一些统计量（例如点击次数，点击次序）来代替。例如一条training data可以是(q, D=[d1,d2,..,dn], S=[c1,c2,..,cn])，其中ci代表di被查询者点击的次数。

学习排序的每一条testing data形如(q,T=[t1,t2,..,tm])，其中q为query，T为候选documents的集合。如前所述，学习排序的训练目标是获得最优的排序模型f(q,d)。当训练完成并进入inference阶段时，排序模型会对每一对候选的(q,d=ti)进行打分，从而得到相关度从高到低的排序。

## 排序模型的评估
排序模型的评估标准需要满足几个要求：
>* 倾向于选择能把相关度高的document排在前面的排序模型。
>* 当列表中documents顺序不对的时候，也能对列表的排序做出合理的打分。例如已知ground truth是，某个query对应的top3 documents分别是d1,d2,d3，那么对于d1,d3,d2和d3,d2,d1这两个排序，能判断出前者优于后者。
>* 要考虑到不同位序的影响权重的不同。例如，假如最终排序结果是长度为100的document列表，那么前10个documents当中出现错序要比第91到第100个document当中出现错序更加严重。
>* 要便于计算。

排序学习中常用的评估准则有DCG，NDCG，AUC，MAP(Mean Average Precision)等等
### DCG(Discounted Cumulative Gain)

### NDCG(Normalized DCG)

### AUC(Area Under ROC)


根据损失函数定义方式的不同，L2R可以分为三类：Pointwise方法，Pairwise方法，Listwise方法
## Pointwise

## Pairwise

## Listwise

[1]: http://times.cs.uiuc.edu/course/598f14/l2r.pdf
[2]: https://tech.meituan.com/2018/12/20/head-in-l2r.html
[3]: https://quinonero.net/Publications/predicting-clicks-facebook.pdf
