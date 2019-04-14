## 常用的召回算法

### 1. UserCF 
即基于User的协同过滤（Collaborative Filtering）。UserCF的思路是向目标用户推荐跟他兴趣相似的用户所喜欢的item。所以UserCF通常包含两个过程：
#### 1).找到跟目标用户兴趣相似的用户集合
#### 2).找到该集合中的用户喜欢的，而目标用户尚未接触过的item并推荐给目标用户

用户之间的similarity是通过用户的行为统计得到的。这里的行为指的是用户对item的点击、浏览、购买、评分等行为（通常点击、浏览被称为隐性行为，这类行为携带信息量小，但由于发生频率高，数据量大，因而也具有统计价值；购买、评分被称为显性行为，这类行为更直接反映了用户喜好，但是发生频次较低，数据规模较小）。例如基于点击行为计算用户的Jaccard similarity：
>* 有两个用户user1和user2，他们各自点击过的item集合分别是S1和S2，那么user1和user2的Jaccard similarity可以定义为S1和S2交集的模除以S1和S2并集的模。

similarity的定义可以有多种方式，除了上例采用的Jaccard similarity之外，还可以采用Cosine similarity，Pearson similarity等等。

user CF的计算方式如下（以点击行为为例）：
>* 每个数据样本形如(user, item)，即某个user点击了某个item。为简便，同一个user对同一个item的多次点击只算做一次。即点击行为为按bool值计算，true代表点击过（无论点击了多少次），false代表没点击过
>* 假设有m个user，n个item，总点击数为N，即每个item平均被N/n个user点击过，每个user平均点击了N/m个item
>* 首先要得到字典形式的item-user关系表dict={item:users}，dict的key是每一个item，value是点击过这个item的所有user的列表
>* 然后遍历item-user关系表求得user-user相似度矩阵U。矩阵U为m * m维，U每一个元素Uij代表第user-i和user-j共同点击过的item总个数
>* 计算字典形式的user-item关系表dict={user:items}，dict的key是每一个user，value是这个user点击过的所有item的列表
>* 得到U和user-item关系表之后，就可以方便的求出任意目标user的相似user集合，然后推荐这个集合当中的user所点击过的item给目标user
>* 从上可以看出，要想给一个user做推荐需要先计算得到user-user相似度矩阵U和user-item关系表。建立矩阵和关系表的时间复杂度都为O(N)，空间复杂度分别为O(m * m)和O(N)

UserCF的可解释性较好，推荐理由可以是“跟你兴趣类似的人也喜欢”这种。

### 2. ItemCF
ItemCF的思路是给用户推荐跟他之前喜欢的item类似的item。对于一个item i，找到与其相似的一个item集合I，然后把I推荐给喜欢i的人。这里item之间的相似度是通过user对它们的行为来判定的，itemCF假设，如果有很多人同时喜欢item a和item b，那么说明a和b具有较高的similarity。

itemCF的计算类似userCF，只不过需要计算的是item-item相似度矩阵而非user-user相似度矩阵。itemCF优于userCF的地方：

>* item-item相似度比user-user相似度更稳定，因为物品的属性相对规定，行为分布也相对稳定（比如90%的好评率），而人的口味会变，行为随机性强
>* 同一个用户可能只有几十个几百个行为记录，同一个item有几万几十万行为记录；因此基于item算相似度更准确
>* 用户数目可能几千万甚至上亿，而同类型的item种类少得多，因此item-item相关矩阵比user-user相关矩阵小得多，基于item算相似度更经济
>* 由于item特性相对固定，可以预先计算item-item相关矩阵
>* itemCF的可解释性更好，推荐理由可以是”类似商品“、“喜欢a的人也喜欢”、”购买了a的人也买了“这种

itemCF也有缺点，比如缺乏新颖性，时常会推荐一些重复雷同的商品。

itemCF和userCF有很多共同的特点，比如都没有学习过程，不需要迭代；都需要遍历样本集，构造相似度矩阵和相关表；都不需要user和item的属性特征；都可以根据新增样本进行增量更新；都有冷启动的问题；都需要大量的数据样本才能捕获user或item的相似度关系并得到较好的推荐效果。

### 3. 隐语义模型（Latent Factor Model）
在itemCF或userCF中，item和user都被表征为高维one-hot vector。one-hot vector是无法直接计算similarity的，因而itemCF和userCF都采用了”曲线救国“的方式计算similarity：把每个item表征成一组users，在user空间计算item的similarity；或者把每个user表征成一组items，在item空间计算user的similarity，两个空间通过用户的行为联系起来。因此itemCF和userCF都被称为是基于邻域的推荐。

而隐语义模型采用了不同的思路，它借鉴了NLP中的embedding的思想，把item和user同时从高维稀疏one-hot vector转化为低维稠密latent vector，相当于把item空间和user空间投影到同一个低维度latent空间，从而可以直接计算item-item、user-user、item-user的similarity。
在隐语义模型中，每个训练样本形如：(user, item, label)。label既可以是bool即{点击过，没点过}，也可以是int即numOfClicks。隐语义模型的训练过程如下：
>* 首先将每个user和每个item都随机初始化为K维向量即latent factor，这里K是超参数，即latent factor的维度，一般是几十，通常不超过100。模型的参数即所有user和所有item的latent factor。
>* 开始迭代。每一轮迭代中，对于每一个训练样本，用user latent factor和item latent factor的内积去逼近label。迭代通常采用SGD的方式来调整参数降低loss。
>* 迭代结束时，算法最终输出就是所有user latent factor和item latent factor。任意两个user或item的相似度都可以从对应latent factor的内积得到。

假设总共有N个训练样本，m个user，n个item，迭代L次，那么时间复杂度为O（L * N * K），空间复杂度为O（m * K + n * K）。

也可以从矩阵分解的角度理解隐语义模型。user和item的行为关系可以用m * n维的矩阵R表示，Rij表示第i个user对第j个item的行为。隐语义模型试图将user表征成m * K维矩阵M，item表征成n * K维矩阵N，并用M * N逆去逼近R。 

隐语义模型有很多变体，例如spark.ml里的ALS算法（Alternating Least Squares）就是隐语义模型的一种分布式实现。

隐语义模型可以方便的找到每个用户的topN like，item的topN similar，然后离线写入数据库或内存，以便后续在线查找。由于不需要user或item相关矩阵，隐语义模型对存储的需求远小于itemCF或userCF。

隐语义模型的缺点是可解释性较差；另外对于新增的样本没办法增量计算，只能从头批量计算。

### 4. Personal Rank
Personal Rank是一种基于二分图的模型，该模型的提出是受到了著名的Page Rank算法的启发。图中有两类顶点，分别是user顶点和item顶点，图中的每一条边都连接了一个user顶点和一个item顶点。如下图，黑色圆代表user，白色方框代表item，user和item之间的边代表该user对该item产生过行为。
![Image text](https://github.com/pengxiaoo/recommender-system/blob/master/imgs/personal_rank.png)
在Personal Rank算法中，找到一个user的topN like，就是找到跟user不直接相连，并且相关度最高的N个item顶点。
图中顶点的相关度主要取决与以下因素： 
>* 两个顶点之间路径数(越多越好) 
>* 两个顶点之间路径长度(越短越好) 
>* 两个顶点之间路径经过的顶点的出度(越少越好) 

Personal Rank通过随机游走的方式来确定user顶点和item顶点之间的相关度。假设给用户A进行个性化推荐，从图中某个user顶点VA开始游走，游走到一个节点时，首先按照概率alpha决定是否继续游走，还是停止这次游走并直接返回VA顶点开始重新游走。如果决定继续游走，那么就从当前顶点出度中的顶点中按照均匀分布随机选择一个作为下次经过的顶点。这样经过很多次的随机游走后，每个item顶点被访问到的概率就会收敛，这个概率就是该item和user的相关度。最终推荐列表中物品的权重就是item节点的访问概率。上述描述可以用公式表示为：
![Image text](https://github.com/pengxiaoo/recommender-system/blob/master/imgs/personal_rank_formula.png)
上图中，PR(v)代表顶点v的访问概率。in(v)和out(v)分别代表顶点v的入度和出度。

Personal Rank对每一个user进行推荐都要全图迭代，比较耗时。工业界应用时通常是采用矩阵运算方式，对所有user同时进行迭代。

注：上面的示例图来自于项亮的《推荐系统实践》一书。

### 5. item2vec
item2vec是一篇2016年的论文[item2vec: neural item embedding for collaborative filtering][1]提出的一种召回算法。它针对电商购物场景，模仿NLP里著名的word embedding算法[word2vec][2]，把每个订单对应的购物篮中的所有items视作一个item set，这个item set就相当于word2vec里的一个context window，item set当中的items相当于位于同一个context window内的words。word2vec的目标函数是给定center word下最大化context words的条件概率，而item2vec的目标函数则是给定item set下最大化不同items共现的条件概率。除了目标函数稍有不同之外，item2vec和word2vec的训练方式几乎是一样的，可以直接用现有的word2vec工具包来训练item2vec。论文中拿item2vec和一种基于SVD分解的隐语义模型做性能对比，item2vec在准确率上更胜一筹。

item2vec看上去比较酷炫，本质上并没有太多创新，只是把word2vec套用了一下，把item从高维one-hot vector embedding成低维稠密vector。不过item2vec可以直接在word2vec工具包上训练，不需要太多额外开发工作，这是其一大优势。item2vec适用范围较窄，主要适合的场景是电商中的购物篮分析。

[1]:https://arxiv.org/pdf/1603.04259.pdf
[2]:https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

### 6. content-based recall
除了上述几种召回算法之外，还有一种重要的召回算法叫做基于内容特征的召回。基于内容的召回的思路是：
>* 对user和item分别做feature提取，离线建立user->feature映射表(或者叫正排表或正排索引)和feature->item映射表（或者叫倒排表或倒排索引），加好索引，然后放入数据库或内存
>* 如果要对某个user做推荐，在线读取user->feature映射表和feature->item映射表，直接构造user->item形成推荐结果

不同于之前的召回算法，content-based需要对user进行画像，对item进行特征提取，因此需要较多的数据预处理和特征工程工作。user画像既包括对user浏览点击历史的统计，也包括用户本身的固有属性，例如用户的年龄，性别，职业等；对item的特征提取，主要包括topic和genre的finding，给item打上标签。以文本类的item为例，特征提取通常分为两类：
>* 基于keyword的特征提取。包括vocabulary的建立，Named Entity识别，keyword排序，user->keyword映射表和keyword->item映射表的建立。
>* 基于topic的特征提取。通过提取文本内容的topic，对不同的item按照topic分类，并根据用户的浏览习惯给用户打上感兴趣的topic的标签，把用户和item通过topic联系起来。提取文本topic有多种方式，传统方式有LDA（Latent Dirichlet Allocation），LSA(Latent Semantic Analysis)，现在比较潮的多是基于embedding方式来提取文本的topic，例如word2vec/doc2vec/deep auto encoder等等。值得一提的是，在不同的文献中terminology略有不同，topic有的地方也被叫做concept或者latent factor。

基于内容召回需要注意的是，对user的feature提取要考虑时间衰减，这是因为user的兴趣和习惯会随着时间发生变化，user近一个月的点击浏览数据会比一年前的点击浏览数据更重要。

基于内容召回的特点：能较好的处理冷启动，新增的item只要有了特征提取就可以推荐出去，新增的user只要有一些固有属性的画像就可以获得推荐；user的推荐结果只跟该user的画像有关，不受别的user的影响；对特征工程有较高要求，需要domain knowlege才能做好user画像和item特征提取。

基于内容的召回在信息流推荐场景中非常重要，因为新闻类item的时效性极强，新收录的item迫切需要被推出去，过一两天再推效果就会大打折扣。以头条系为例，基于内容的推荐是其推荐系统的核心，头条投入了大量的人力去做特征工程，一个主要原因就是为了做好基于内容的推荐。
