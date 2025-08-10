# **短信内容分析：垃圾短信还是正常短信 (入门级)**


这是我第一个关于自然语言处理（NLP）的实践项目，我选择了“短信垃圾信息收集数据集”。该数据集包含 5572 条短信文本及其标签，分为“spam”（垃圾短信）或“ham”（正常短信）。

在这个项目中，我将探讨一些常见的NLP技术，例如：

* **移除标点符号和停用词**
* **分词（Tokenizer）、词袋模型（Bag of Words）**
* **词频-逆文档频率（TF-IDF）**

基于这些预处理步骤，我训练了6个不同的模型，用于将**未知**短信分类为垃圾短信或正常短信。

* **朴素贝叶斯分类器 (Naive Bayes Classifier)**
* **支持向量机分类器 (SVM Classifier)**
* **K-近邻分类器 (KNN Classifier)**
* **随机梯度下降分类器 (SGD Classifier)**
* **梯度提升分类器 (Gradient Boosting Classifier)**
* **XGBoost 分类器**

为了简化对训练和测试数据的预处理流程，并方便在相同条件下优化不同模型，所有分类任务都通过包含`GridSearchCV`的**管道（Pipelines）**来完成。最后，我们使用多种**评估指标**来衡量模型性能，包括：准确率（accuracy）、精确率（precision）、召回率（recall）、F1分数（f1-score）和ROC曲线下面积（roc_auc）。

---

## **项目大纲**

* [**第 0 部分：导入与准备工作**](#part-0)
    * 导入所需库
    * 定义常用函数
* [**第 1 部分：探索性数据分析 (EDA)**](#part-1)
    * 数据集概览 (`head`, `describe`, `value_counts`)
    * 目标变量分布
    * 为标签添加数值表示
    * 分析短信长度特征
    * 生成词云 (WordClouds)
* [**第 2 部分：文本预处理**](#part-2)
    * 移除标点符号和停用词
    * 分析正常短信与垃圾短信中的高频词
    * 使用 `CountVectorizer` 构建词袋模型
    * 计算词频-逆文档频率 (TF-IDF)
* [**第 3 部分：构建与评估分类器**](#part-3)
    * 使用训练/测试集进行分割
    * 构建分类管道 (Pipelines) 并使用 `GridSearchCV` 进行超参数调优
    * 比较不同模型的性能指标（混淆矩阵、准确率、精确率、召回率、F1分数、ROC AUC）
    * 针对不同评估指标（精确率、召回率）对模型进行优化
* [**第 4 部分：NLTK 其他功能探索**](#part-4)

---

## **第 0 部分：导入与准备工作**

### **导入库**

```python
# 数据处理
import numpy as np 
import pandas as pd 

# 数据可视化
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# 文本处理与特征工程
import string
import wordcloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# 机器学习模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# Scikit-learn 工具
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 下载NLTK所需数据
# nltk.download('stopwords')
# nltk.download('punkt')
```

### **定义常用函数**

```python
def print_validation_report(y_true, y_pred):
    """打印分类报告和准确率"""
    print("分类报告:")
    print(classification_report(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    print(f"准确率: {acc:.4f}")
    return acc

def plot_confusion_matrix(y_true, y_pred, ax, title):
    """在指定的axes上绘制混淆矩阵"""
    mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5, cmap="Blues", cbar=False, ax=ax)
    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')
    ax.set_title(title, fontweight='bold')

def remove_punctuation_and_stopwords(sms):
    """
    移除文本中的标点符号和停用词，并将所有字母转为小写。
    """
    # 移除标点
    sms_no_punct = "".join([char for char in sms if char not in string.punctuation])
    
    # 分词并转为小写，同时移除停用词
    words = [
        word.lower() for word in sms_no_punct.split() 
        if word.lower() not in stopwords.words("english")
    ]
    
    return words
```

---

## **第 1 部分：探索性数据分析 (EDA)**

### **1.1 数据集概览**

首先，加载数据并进行初步清理。

```python
# 加载数据，注意编码方式
# 假设 spam.csv 文件在当前目录下
try:
    data = pd.read_csv("./spam.csv", encoding='latin-1')
except FileNotFoundError:
    print("错误：spam.csv 文件未找到。请确保文件与脚本在同一目录下。")
    # 创建一个空的DataFrame以避免后续代码出错
    data = pd.DataFrame(columns=['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

if not data.empty:
    # 删除多余的空列，并重命名主要列
    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    data = data.rename(columns={"v1": "label", "v2": "text"})

    # 为标签创建数值列：spam=1, ham=0
    data['spam'] = data['label'].map({'spam': 1, 'ham': 0}).astype(int)

    print("数据前5行:")
    display(data.head())

    print("\n数据集基本信息:")
    data.info()

    print("\n按标签分组统计:")
    display(data.groupby("label").describe())
```

### **1.2 目标变量分布**

数据集包含4825条正常短信（ham）和747条垃圾短信（spam），存在类别不平衡问题。

```python
if not data.empty:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x='label')
    plt.title('短信类别分布 (Ham vs Spam)')
    plt.show()

    print(data.label.value_counts())
```

### **1.3 短信长度分析**

添加一个新特征：短信的字符长度。

```python
if not data.empty:
    data['length'] = data['text'].apply(len)

    data.hist(column='length', by='label', bins=60, figsize=(12, 4))
    plt.suptitle('按类别划分的短信长度分布')
    plt.show()
```

从直方图可以看出，垃圾短信（spam）通常比正常短信（ham）更长。大部分正常短信的长度在100个字符以下，而垃圾短信的长度则多在100个字符以上。这个特征可能对分类有帮助。

### **1.4 词云**
词云可以直观地展示文本中词语出现的频率。

```python
def show_wordcloud(data_subset, title):
    """生成并显示词云"""
    text = ' '.join(data_subset['text'].astype(str).tolist())
    stopwords_set = set(wordcloud.STOPWORDS)
    
    wc = wordcloud.WordCloud(stopwords=stopwords_set, background_color='white',
                               colormap='viridis', width=800, height=400).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc) 
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()

if not data.empty:
    # 分别为正常短信和垃圾短信生成词云
    show_wordcloud(data[data['spam'] == 0], "正常短信 (Ham) 词云")
    show_wordcloud(data[data['spam'] == 1], "垃圾短信 (Spam) 词云")
```

---

## **第 2 部分：文本预处理**

### **2.1 高频词分析**

我们先对文本进行预处理（移除标点和停用词），然后统计正常短信和垃圾短信中出现频率最高的30个词。

```python
if not data.empty:
    # 对原始文本应用预处理函数
    data['processed_text'] = data['text'].apply(remove_punctuation_and_stopwords)

    # 将处理后的词语列表平铺
    all_ham_words = [word for sublist in data[data['spam'] == 0]['processed_text'] for word in sublist]
    all_spam_words = [word for sublist in data[data['spam'] == 1]['processed_text'] for word in sublist]

    # 统计词频
    ham_word_counts = pd.DataFrame(nltk.FreqDist(all_ham_words).most_common(30), columns=['单词', '频次'])
    spam_word_counts = pd.DataFrame(nltk.FreqDist(all_spam_words).most_common(30), columns=['单词', '频次'])

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.barplot(data=ham_word_counts, x='频次', y='单词', ax=axes[0], palette='Blues_d')
    axes[0].set_title('正常短信高频词 Top 30')

    sns.barplot(data=spam_word_counts, x='频次', y='单词', ax=axes[1], palette='Reds_d')
    axes[1].set_title('垃圾短信高频词 Top 30')

    plt.tight_layout()
    plt.show()
```

### **2.2 文本向量化 (TF-IDF)**
机器学习模型无法直接处理原始文本，需要将其转换为数值向量。我们使用 **TF-IDF（词频-逆文档频率）** 方法，它不仅考虑词语在单个文档中的频率（TF），还考虑它在整个语料库中的普遍性（IDF），从而赋予罕见但重要的词语更高的权重。

`scikit-learn` 中的 `TfidfVectorizer` 可以一步完成 **分词、构建词袋、计算TF-IDF** 的所有操作。

---

## **第 3 部分：构建与评估分类器**

### **3.1 划分训练集和测试集**

我们将数据集按照 7:3 的比例划分为训练集和测试集。

```python
if not data.empty:
    X = data['text']
    y = data['spam']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

### **3.2 使用管道（Pipeline）和网格搜索（GridSearchCV）进行模型训练**

为了简化工作流程并系统地寻找最佳模型参数，我们使用 `Pipeline` 将 **文本向量化** 和 **分类器训练** 两个步骤串联起来。然后，`GridSearchCV` 会自动测试不同的参数组合，并通过交叉验证找到性能最好的模型。

下面是我们定义的分类器和它们的参数搜索空间：

```python
# 定义模型和参数网格
classifiers = {
    "朴素贝叶斯 (MNB)": (MultinomialNB(), {
        'clf__alpha': (0.1, 0.5, 1.0),
    }),
    "支持向量机 (SVC)": (SVC(probability=True), {
        'clf__C': [1, 10, 100],
        'clf__gamma': ['scale', 'auto'],
    }),
    "K-近邻 (KNN)": (KNeighborsClassifier(), {
        'clf__n_neighbors': [5, 10, 15],
    }),
    "随机梯度下降 (SGD)": (SGDClassifier(random_state=42), {
        'clf__alpha': [1e-4, 1e-3, 1e-2],
        'clf__penalty': ['l2', 'l1'],
    }),
    "梯度提升 (GBC)": (GradientBoostingClassifier(random_state=42), {
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.1, 0.2],
    }),
    "XGBoost (XGB)": (xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [5, 7],
    })
}
```

#### **优化函数**
我们创建一个函数来自动化训练、评估和存储结果的过程。

```python
results = []

def train_and_evaluate_pipeline(name, model, params, scoring_metric):
    """
    构建管道，使用GridSearchCV训练模型，并评估结果。
    """
    print(f"--- 正在为 '{name}' 训练模型 (评估指标: {scoring_metric}) ---")
    
    # 1. 创建管道
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer=remove_punctuation_and_stopwords)),
        ('clf', model)
    ])
    
    # 2. 创建并运行GridSearchCV
    grid_search = GridSearchCV(pipeline, params, cv=5, scoring=scoring_metric, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # 3. 获取最佳模型和预测结果
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 4. 计算各项指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # 5. 存储结果
    results.append({
        '模型': name,
        '评估指标': scoring_metric,
        '最佳参数': grid_search.best_params_,
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1 分数': f1,
        'ROC AUC': roc_auc,
        '预测结果': y_pred # 存储预测结果以绘制混淆矩阵
    })
    
    print(f"'{name}' 训练完成。最佳分数 (CV {scoring_metric}): {grid_search.best_score_:.4f}\n")

if not data.empty:
    # 使用默认的 'accuracy' 作为评估指标进行第一轮训练
    for name, (model, params) in classifiers.items():
        train_and_evaluate_pipeline(name, model, params, 'accuracy')
```

### **3.3 结果比较**

#### **性能指标汇总**
将所有模型的性能指标整理成一个表格，方便比较。

```python
if results:
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    display(results_df[['模型', '准确率', '精确率', '召回率', 'F1 分数', 'ROC AUC']].round(4))
```

#### **混淆矩阵**
混淆矩阵可以清晰地展示每个类别的预测情况，特别是**假正例 (False Positives)** 和 **假反例 (False Negatives)** 的数量。

* **假正例 (FP)**：将正常短信（ham）错误地识别为垃圾短信（spam）。这可能会导致用户错过重要信息。
* **假反例 (FN)**：将垃圾短信（spam）错误地识别为正常短信（ham）。这会降低用户体验。

对于垃圾邮件过滤系统，我们通常希望**精确率 (Precision)** 尽可能高，以减少 FP 的数量。

```python
if results:
    # 获取第一轮（scoring='accuracy'）的预测结果
    accuracy_preds = {res['模型']: res['预测结果'] for res in results if res['评估指标'] == 'accuracy'}

    # 绘制所有模型的混淆矩阵
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("各模型混淆矩阵 (以 'accuracy' 优化)", fontsize=20, fontweight='bold')
    axes_flat = axes.flatten()

    for i, (name, y_pred) in enumerate(accuracy_preds.items()):
        plot_confusion_matrix(y_test, y_pred, axes_flat[i], name)

    # 隐藏多余的子图
    for i in range(len(accuracy_preds), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
```

从结果来看，**支持向量机 (SVC)** 和 **朴素贝叶斯 (MNB)** 在保持高召回率的同时，实现了非常高的精确率（错误地将正常短信标记为垃圾短信的情况很少），综合表现出色。

### **3.4 针对不同指标进行优化**

#### **以 "精确率 (Precision)" 为目标**
如果我们首要目标是确保不将任何正常短信错判为垃圾短信，那么我们应该以`precision`作为`GridSearchCV`的评估标准。

```python
if not data.empty:
    # 清空之前的非'accuracy'结果
    results = [res for res in results if res['评估指标'] == 'accuracy']

    # 以 'precision' 作为评估指标进行第二轮训练
    print("\n" + "="*50)
    print("  第二轮训练：以 Precision 作为优化目标")
    print("="*50 + "\n")

    for name, (model, params) in classifiers.items():
        train_and_evaluate_pipeline(name, model, params, 'precision')

    # 提取以 'precision' 优化的预测结果
    precision_preds = {res['模型']: res['预测结果'] for res in results if res['评估指标'] == 'precision'}

    # 绘制混淆矩阵
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("各模型混淆矩阵 (以 'precision' 优化)", fontsize=20, fontweight='bold')
    axes_flat = axes.flatten()

    for i, (name, y_pred) in enumerate(precision_preds.items()):
        plot_confusion_matrix(y_test, y_pred, axes_flat[i], name)
    
    for i in range(len(precision_preds), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
```

可以看到，当以精确率为目标时，一些模型（如SVC）在测试集上达到了100%的精确率，这意味着它们预测为“垃圾短信”的短信中，没有一个是正常的。

---

## **第 4 部分：NLTK 其他功能探索**

NLTK库还提供了许多其他有用的文本处理功能。

### **分句 (Sentence Tokenization)**

```python
if not data.empty:
    sample_text = data['text'][7]
    print("原始文本:", sample_text)

    # 将文本分割成句子
    sentences = sent_tokenize(sample_text)
    print("\n分句结果:", sentences)
```

### **分词 (Word Tokenization)**

```python
if not data.empty:
    # 将文本分割成单词
    words = word_tokenize(sample_text)
    print("\n分词结果:", words)
```