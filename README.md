# IJCAI-18 阿里妈妈搜索广告转化预测   
    经过不算短的时间，IJCAI-18比赛结束了，最后我们队伍也只是排在了81名。在比赛过程发现了自身的很多不足，还是需要继续向大家学习。感谢我的队友小伙伴，互相学习是一件很爽的事情。  
    这里，对我个人的算法进行开源（包括了队友的一些特征），也希望大家可以从我简陋的代码中学到一些东西。   
    以下以复赛为例。  

---
## 数据   
    data目录下保存原始文件，由于文件比较大的原因，就没有进行上传。   
    对于数据的基本理解可以看天池的赛题与数据，下面讲一下我自己的思路和想法。    
    线下训练集：6号全天数据，7号0-11点数据；线下测试集：7号11-12点数据。    
    线上训练集：6号全天数据，7号0-11点数据；线上测试集：7号12-24点数据。    
    当换到复赛时，发现线下训练集不像初赛的时候比较好选。最后通过观察线上与线下的增减关系选择了如上所示的线下测试集（来自于队友），通过统一的线下测试集，方便了我们最后进行简单加权融合时计算系数。     
    通过对各个特征进行value_counts()，发现user_id，item_id，shop_id都存在一些数据量很少的字段。为了便于训练和解决数据量出现次数少的问题，我设置了阈值来进行筛选，将低于该阈值的id特征改为同一值。（现在来想，是不是应该先清洗一下数据，删除掉浏览次数多，但是实际购买少的id特征）   


---
## 特征   

    Features目录下为提取特征的文件，self来自于我，其他来自于队友。特征构建发生在全部数据中。   
    [1]提取时间特征：天，小时，分钟。     
    [2]提取关于时间的统计特征：通过不同的id特征来进行分组，针对不同的时间来提取size(),count(),cumcount(),unique()等特征。       
    [3]提取关于时间的间隔特征：通过不同的id特征来进行分组，提取上/下次访问/购买时间差。   
    [4]提取关于时间统计特征的比值特征：在相同的id特征分组得到的统计特征后，在不同时间区域上进行比值。      
    [5]提取转化率特征（来自于队友）。      
    [6]提取相似度特征（来自于队友）。    
    [7]其他想不起来的特征=-=。    
    最后得到了300+维特征，但是有70+特征在lightgbm模型中特征重要性为0。   
    
---
## 模型   

    python目录下为对数据的预处理，特征合并以及lgb模型预测的py文件。       
    模型使用的是lightgbm模型，也尝试过其他的模型deepFM,NN模型，但是由于个人不太会用的原因，效果不好。     
    使用lightgbm模型在上述特征下进行预测，在TestB上分数为0.14058。     
    在item_id上进行0.8采样5次分别使用lightgbm训练，均值后，在TestB上分数为0.14050。    

---
## 技巧   

    [1]可以通过提交相同值，根据得分来知道is_trade中有多少个1，由此可以计算平均值。     
    [2]在知道真实平均值的情况下可以通过修正来使预测的均值逼近真实平均值。来源于：https://github.com/infinitezxc/kaggle-avazu      
    
---
## 缺点   

    [1]在特征工程上，还是有很大的欠缺，单模型与前排大佬差距还是很大。而且有大部分特征在lightgbm模型中是无用的，但是没有进行过特征选择。在此过程中，心情还比较浮躁-=-      
    [2]最终只有一个lgb模型，只能在lgb模型中进行融合，导致融合的提升并不大，还要多多学习！       
    
    
    以上！
    
