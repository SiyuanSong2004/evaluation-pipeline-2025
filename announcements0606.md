# 0606公告1：【关于最终评测】

各位好！
距离模型提交截止（6月11日）还有几天。现将评测结果的提交流程说明如下，请仔细阅读。

重要：本次比赛所有deadline的时区统一为UTC-12（AoE），即anywhere on earth23:59。

## 第一阶段：最终模型上传（6月11日前）

请各队务必在6月11日23:59 (AoE)前，将本团队模型上传至Hugging Face。

使用6月11日之后上传的模型得到的测试结果视为无效。

需上传的有：
1. 预训练的模型
2. config.yaml：之后不能调整

**即6月11日23:59以后，只能跑一遍官方最新pipeline.py+你的config.yaml，评测模型，不能预训练新模型。**

你可以训练多个预训练模型，只要在6月11日23:59 (AoE)前上传到Hugging Face即可参与最终评测。

## 第二阶段：测试隐藏任务、提交结果（6月12日–20日）

6月12日主办方将公布隐藏任务。

隐藏任务包括：
1. 一个minimal pair任务，评测方法与ZhoBLiMP相同，使用预训练模型
2. 一个中文NLI任务，评测方法与OCNLI相同
3. Word fMRI和fMRI更多被试的数据
4. 额外的汉字评测（minimal pair格式），评测方法与现有汉字评测相同，使用预训练模型

同时将发布最终版pipeline并更新leaderboard，请你下载隐藏任务数据至指定文件夹，运行pipeline+你的config进行最终评测，并上传结果。

最终成绩为：所有公开任务与隐藏任务上得分的平均值。

请各队在6月12–20日期间进行评测，并通过Hugging Face leaderboard提交结果。提交需包含：
* 队名（须与报名时一致）
* 提交者姓名与所属组织
* 模型名
* Hugging Face模型ID
* 联系方式

请注意：只有在6月12日leaderboard更新之后提交到leaderboard的结果才视为有效（更新前的旧提交仅作参考，不计入）；所有有效结果须经主办方验证。使用leaderboard过程中如遇问题，请随时联系主办方。

提交结果截止：6月20日23:59 (AoE)

## 第三阶段：获奖验证及模型和评测报告撰写（6月20日后）

结果提交截止后，主办方将联系可能获奖的队伍，请其提交训练数据与源代码以供复现验证。
按以往NLPCC惯例，主办方邀请排名靠前的队伍撰写模型和评测报告。

其余细节以官网guideline页面为准：https://chinese-babylm.github.io/guidelines.html
本说明也将同步更新至官网。


# 0606公告2【关于汉字评测】

一位参赛同学提出原先的汉字评测中，有很多不太常见的偏旁部首如艹，会导致某些分词器出现较多UNK token。

经过讨论，主办方做了两个改进：

1. 更新了汉字评测中的两个benchmark，确保：

a. 目标汉字和目标偏旁部首均为3500常用字表中的汉字。
b. 目标汉字和目标偏旁部首在官方公布的预训练数据中至少出现了50次。

例："“伴”和“办”的声母、韵母和声调完全相同。"      其中伴和办均为常用字，且至少出现了50次。
例："“汞”字上边的部分是“工”。"    其中汞和工均为常用字，且至少出现了50次。

此方法最大程度减少了UNK的出现。

2. 汉字评测的指标：

凡一个或多个目标汉字或目标偏旁部首为UNK token的模型，该题目不得分。评测指标仍为accuracy。

例：一共有1000个题目，一个系统在500个题目中，至少一个目标汉字或目标偏旁为UNK，即不被该系统的分词器识别，则该系统最高acc为0.5。若该系统在剩下的500个题目中（没有任何UNK），答对了300个题目，即good_sent的logprob大于bad_sent，则该系统最终acc为0.3。

请各位参赛队伍在更新的汉字评测上进行测试。更新后的数据地址不变：
https://huggingface.co/datasets/chinese-babylm-org/hanzi-pinyin
https://huggingface.co/datasets/chinese-babylm-org/hanzi-structure

可以通过拉取最新代码，运行：python pipeline.py download --tasks hanzi_structure hanzi_pinyin --force-download 下载更新后的数据

# 0606公告3【pipeline更新】

1. 更新了汉字评测的数据。
2. 更新了pipeline.py，用新方法计算汉字评测得分。
3. 更新了config.yaml
- 可以添加：save_item_with_unk: true，记录汉字评测中包含UNK的题目，供debug
- 可以针对NLU中微调任务设置不同超参数


