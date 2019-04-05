# recommender-system

推荐系统的目标是向user推荐他可能感兴趣的item。在早期，推荐问题常常被看做评分预测问题，即已知user对一组item的评分，预测他对一个新的item的评分。著名的netflix竞赛就是属于这种类型。近年来工业界推荐系统更侧重于topN推荐，即向user推荐他最可能感兴趣的N个items。topN推荐的损失函数定义和评估体系都和评分预测大有不同。topN推荐在实际生活中的应用更广泛，本文所针对的都是topN推荐。

推荐系统整体上可以分为前端和后端两大部分。前端包括App或网页上呈献给用户的推荐位、推荐条目信息，以及对用户点击浏览行为的埋点记录；后端包括推荐算法和推荐效果评估。前端和后端会形成闭环：推荐算法->app上对用户呈现->用户行为的记录和反馈->推荐效果评估->改进推荐算法。

从功能角度，推荐算法可以分为两大部分：召回(Recall)和排序（Rank）。所谓召回，是指从item全集中筛选出推荐候选集；而排序则是指对候选集中的每一个item进行匹配度打分，将items按照匹配度得分进行降序排列。召回通常是离线跑批方式运行，而排序通常需要针对用户请求进行实时响应。

在十年前，人们对于推荐算法的研究侧重于召回算法。常见的召回算法有协同过滤、隐因子模型、基于图的召回算法等等。关于召回算法的研究资料比较丰富，其中值得推荐的入门资料是项亮的《推荐系统实践》一书。近年来随着手机成为最主要的信息传播媒介，同时不同应用之间的竞争愈发激烈，人们对推荐系统提出了更高的要求。为了进一步挖掘推荐系统的性能，业界对排序算法的关注和研究在逐渐增加。

下面分别从召回和排序两个方面介绍推荐算法。

## 召回
### 常用的召回算法
召回算法是推荐系统当中研究和应用比较成熟的领域。[常用的召回算法][1]总结了业界应用比较广泛的几种召回算法。

### 召回算法小结
召回算法有很多，除了上述提到的几种之外，还有基于用户调研的召回，基于社交网络的召回（例如”你的好友XXX也买了这本书“）等等。不过本质上与前面介绍的召回算法并没有太大差异，这里不再单独细述。如以基于社交网络的召回为例，其主要过程是建立和查询user->friend映射表以及friend->item映射表，与content-based推荐很类似。

召回算法的主体部分通常是以离线方式运行分布式任务（例如spark）来完成的。一个典型的spark任务通常包含：
>* 读取业务数据库和log数据库，进行数据预处理
>* 生成user画像和item特征、user和item之间的关系，有时甚至直接生成候选集
>* 对生成好的特征和候选集建立索引，写入数据库（如redis）

对于离线生成的特征如何高效的建立索引，是召回算法至关重要的一环，索引的查询性能直接影响了后续排序算法的耗时。索引通常分两大类：一类是B树，这也是大多数数据库内建的索引方式，对于结构化、半结构化的数据非常有效；另一大类是近似最近邻哈希算法（Hashing algorithms for Approximate Nearest
Neighbor Search），对于非结构化的数据例如文本、图片非常有效。根据数据类型的不同，近似最近邻哈希也有不同的实现方式，例如geo信息可以通过GeoHash、文本可以通过SimHash、图片可以通过CNN bottleneck layer的encoding并结合LSH(Locality Sensitive Hashing)来实现。

在工业界实际应用中，很少单独采用一种或两种召回算法，通常采用多种召回算法并行的架构，每种召回算法独立运行产生召回结果，最后merge起来然后喂给Rank模块。

## 排序
所谓推荐系统中的排序，是指对于召回的候选集中的item，根据其属性，并结合用户画像以及上下文信息，计算出匹配程度的得分。items之间的得分高低就反映了它们不同的呈现优先级，等同于进行了排序。

从点击率预估的角度理解，Rank就是预测给定user和上下文情况下，每个item的点击率。

### 排序算法的输入和输出
从上面的描述可知，排序算法的输入是召回的候选集+上下文信息+用户画像，输出是给候选集中的每个item赋予的打分。召回候选集无须赘述，上下文信息是指app用户进入特定页面，发起推荐请求时所携带的动态信息，比如时间、地理位置、页面类型、请求的item类型等；用户画像是指用户的静态信息例如年龄、性别、职业等，以及用户的行为统计特征例如历史点击浏览记录、购买记录等。上下文信息是动态和瞬时的，实时获取，并在实时计算中使用；而用户画像是相对静态和稳定的，通常是预先离线跑spark任务计算好并放入数据库（是召回算法的工作的一部分），以便排序时实时查询。

### 排序算法的架构
与主要以离线方式运行的召回算法不同，排序算法基本是实时的。在实际应用当中，排序算法通常作为一个服务（Rank as a Service）供推荐系统调用。Rank Service的核心是打分函数，打分函数可以根据召回候选集+上下文信息+用户画像，对每一个item打分。打分函数可以由人工规则制定，也可以是机器学习学出来的。如果是前者，Rank Service直接调用打分函数对每个item打分；如果是后者，Rank Service会加载预先离线训练好的model对每个item打分。

在实际的推荐系统中，排序通常有不止一轮。以常见的三轮排序为例，分别是：
>* 第一轮粗排序，主要目的是对输入的候选集进行筛选，控制进入主排序的item数目。粗排序处理的item数量通常比较大，因此往往采用一些简单的规则进行排序筛选。粗排序的例子比如LBS相关应用中，根据用户的地理位置，把10公里之外的商户先过滤掉，只保留用户附近的商户。
>* 第二轮主排序，是最核心的排序算法，很大程度上决定了最终返回结果。主排序的得分函数通常是机器学习的方式训练得到的，即后面会讲到的Learning to Rank。
>* 第三轮针对topN进行精细排序和调整。在移动互联网的时代，用户在第一屏没看到自己感兴趣的内容，有可能就直接退出app，因此一个推荐系统的成败很大程度上取决于前30条甚至前10条推荐。因此在最终返回给用户前，根据用户画像和上下文信息，对topN的item进行精细Re-Rank，逐渐成了业界普遍采用的做法。

### 常用的排序算法
按照打分函数的制定方式，排序算法可以分两类：人工规则排序和机器学习排序(Learning to Rank)。人工规则排序需要较多的专家经验和领域知识，随着特征的增多，打分函数的设计会越来越困难，很容易过拟合。如今机器学习排序已经成为业界的主流。[常用的机器学习排序算法][2]对机器学习排序做了总结。

## 推荐系统评估
从评估方式的角度，推荐系统的评估可以分为两类：
>* 离线评估：通过监督学习的方式，从历史真实数据（item全集，user全集，user-item行为历史的日志）抽取训练和测试样本，通过一些离线评估指标对算法评估；
>* 在线评估：通过ab-test，评价算法对线上业绩指标（以新闻类推荐为例，线上业绩指标可以是点击率和平均阅读时长；以电商为例，线上业绩指标是下单量和成交额）的改进效果。
[推荐系统评估][3]介绍了这两种评估方式。

[1]: https://github.com/pengxiaoo/recommender-system/blob/master/docs/recall.md
[2]: https://github.com/pengxiaoo/recommender-system/blob/master/docs/rank.md
[3]: https://github.com/pengxiaoo/recommender-system/blob/master/docs/evaluation.md
