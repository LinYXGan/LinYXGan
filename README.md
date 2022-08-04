# 算法说明
## 简介
- 赛事名称：疫情微博情绪识别挑战赛
- 赛事时间：6月24日——7月23日
- 队伍名称：随便打打
- 队伍成员：QNQvyURr

## 算法主要流程
- 将数据中表情转变为文字，去除用户名，去除无意义符号等。
- 采用预训练模型+微调的方式搭建模型。
- 采用Mutli-dropout与dym特征池化。
- 采用五折交叉验证微调模型。
- 采用硬投票方式进行模型融合。

还尝试对抗训练以及测试集伪标签等方法，无效果，因此不采用。其中对抗训练在不使用dropout时会有一定提升，但效果不如使用Mutli-dropout。
同时因个人设备限制，五折交叉验证训练中每折只训练了1个epoch，如果增加至3个epoch效果理论上会更好。
## 模型
依据个人经验以及一定的实验，最终采用了ernie-3.0-base、ernie-3.0-xbase、hfl/roberta-wwm-ext-large、nezha-large-wwm-chinese四种预训练模型。

## 环境
基于paddle2.3.1框架，在飞桨AI Studio平台，采用V100(32GB)GPU、32G内存环境进行模型训练。
库：os、json、random、time、numpy、pandas、tqdm、sklearn、paddle、paddlenlp、paddle.nn、functools、matplotlib、seaborn、scipy、emojiswitch、re。
## 效果
- ernie-3.0-base线上可达0.967，加入Mutli-dropout与dym特征池化最高可达0.9735，五折交叉验证无提升。
- ernie-3.0-xbase线上可达0.967，加入Mutli-dropout与dym特征池化可达0.972，五折交叉验证效果可达0.9729。
- hfl/roberta-wwm-ext-large与nezha-large-wwm-chinese线上可达0.965，加入Mutli-dropout与dym特征池化可达0.97，五折交叉验证效果可达0.9719。
- 四类模型进行融合硬投票(2个ernie-base模型，其它模型各一个)线上效果可达0.9740。
- 采用多组硬投票组合进行投票后平均，最终线上效果为0.9748。
且投票模型融合中，初次模型融合选择相互之间结果差异较大的模型来融合，效果较好。
## 代码说明
ernie3.0_base.ipynb文件为基础的单模型训练程序。带有"5folds"后缀文件为5折交叉验证训练程序，如Roberta_5folds为roberta模型的五折交叉验证训练程序。Voting为模型融合投票程序。
操作时先运行5个模型训练程序，再将结果进行融合。而采用多组模型融合进行最终投票，则需要训练多组模型，此时需将模型训练文件中seed进行改变，增大模型之间的差异。

## 最后感谢科大讯飞xDatawhale提供的这次比赛机会。感谢举办方提供的比赛baseline。
