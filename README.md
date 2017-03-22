# Df_sougou
大数据精准营销中搜狗用户画像挖掘<br>
队伍名:我很抱歉<br>
排名:初赛A榜 133/983<br>
<br>
# 预处理<br>
jieba分词，然后根据停用词表去掉高频率词<br>
<br>
# 特征工程<br>
词袋模型，TF-IDF加权，chi-square选取前二十万维特征（尝试过对词做组间组内方差加权，效果不好）<br>
<br>
# 模型<br>
线性SVC,BernoulliNB,MultinomialNB,逻辑回归四模型分别预测然后Voting<br>
（SVC线性核可以到前五十，但是我的特征做得不好，导致SVC效果没有朴素贝叶斯好。特征还可以有很多优化。）<br>
<br>
# 模型融合<br>
看了前五的代码，原来是用Stacking融合多个模型，我只用个了原始的Voting...<br>
要好好学一下集成学习
