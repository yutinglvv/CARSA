# 导入一些必要的库
import numpy as np
import networkx as nx
import simhash
import json
import nltk
from flask import Flask, request, jsonify

app = Flask(__name__)

# 定义一些常量
NUM_KEYWORDS = 10  # 查询关键词的数量
NUM_MASHUPS = 100  # Mashup数据集的大小
NUM_APIS = 500  # API数据集的大小
NUM_RESULTS = 10  # 推荐结果的数量
THRESHOLD = 3  # Simhash去重的阈值

mashup_dataset = []
api_dataset = []

from gensim.models import KeyedVectors

glove_input_file = 'glove/glove.6B.50d.txt'  # 请根据您下载的文件名进行修改  
model = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)


# 定义一些辅助函数
# 获取开发者输入的查询关键词，返回一个字符串列表
def get_query_keywords(query_word):
    # 这个函数可以根据你的具体需求来修改，这里只是一个简单的例子
    # # 假设开发者输入的是一个字符串，用空格分隔关键词
    query = query_word
    # 将字符串分割成列表，并去除空白字符
    keywords = query.split()
    keywords = [k.strip() for k in keywords if k.strip()]
    # 返回关键词列表
    return keywords


# 获取Mashup数据集，返回一个字典，键是Mashup的ID，值是Mashup的描述和使用的API列表
def get_mashup_dataset(path):
    # 这个函数可以根据你的具体数据来源和格式来修改，这里只是一个简单的例子  
    # Mashup数据集是一个JSON列表，存在名为"active_mashups_data.txt"的文件中  
    with open(path, "r", encoding='utf-8') as f:
        json_list = json.load(f)

        # 创建一个空字典
    mashup_dataset = {}

    # 遍历JSON列表  
    for item in json_list:
        # 获取Mashup的ID，描述和使用的API列表  
        mashup_id = item['title']
        mashup_description = item['description']
        try:
            mashup_apis = [api['title'] for api in item.get('related_apis', []) if
                           api] if 'related_apis' in item else []
        except Exception as e:
            print(item)

        # 将Mashup的信息存入字典中  
        mashup_dataset[mashup_id] = {"description": mashup_description, "apis": mashup_apis}

        # 返回Mashup数据集字典
    return mashup_dataset


# 获取API数据集，返回一个字典，键是API的ID，值是API的描述
def get_api_dataset(path):
    # 这个函数可以根据你的具体数据来源和格式来修改，这里只是一个简单的例子  
    # API数据集是一个JSON列表，存在名为"active_apis_data.txt"的文件中  
    with open(path, "r", encoding='utf-8') as f:
        json_list = json.load(f)

        # 创建一个空字典
    api_dataset = {}

    # 遍历JSON列表  
    for item in json_list:
        if item == None:
            continue
        try:
            # 获取API的ID和描述  
            api_id = item['title']
            api_description = item['description']

            # 将API的信息存入字典中  
            api_dataset[api_id] = {"description": api_description}
        except Exception as e:
            print(item)

    # 返回API数据集字典  
    return api_dataset


# 计算两个Mashup之间的相似度，返回一个浮点数
def get_mashup_similarity(m1, m2):
    # 这个函数可以根据你的具体相似度度量方法来修改，这里只是一个简单的例子# 假设两个Mashup之间的相似度由它们使用的API之间的相似度决定，即Jaccard相似系数
    from scipy.spatial.distance import jaccard
    # 获取两个Mashup使用的API列表
    apis1 = mashup_dataset[m1]["apis"]
    apis2 = mashup_dataset[m2]["apis"]
    # 计算Jaccard相似系数，并转换为相似度
    jaccard_distance = jaccard(apis1, apis2)
    jaccard_similarity = 1 - jaccard_distance
    # 返回相似度
    return jaccard_similarity


# 计算两个API之间的相似度，返回一个浮点数
def get_api_similarity(a1, a2):
    # 这个函数可以根据你的具体相似度度量方法来修改，这里只是一个简单的例子# 假设两个API之间的相似度由它们的描述之间的语义相似度决定，即余弦相似度
    from scipy.spatial.distance import cosine
    from gensim.models import Word2Vec
    # 加载一个预训练的词向量模型，这里使用Google News的模型，你也可以使用其他的模型
    # model = Word2Vec.load("GoogleNews-vectors-negative300.bin")

    # 获取两个API的描述
    description1 = api_dataset[a1]["description"]
    description2 = api_dataset[a2]["description"]
    # 将描述分词，并过滤掉停用词和标点符号

    from nltk.corpus import stopwords
    from string import punctuation
    nltk.download("punkt")
    nltk.download("stopwords")
    tokens1 = nltk.word_tokenize(description1)
    tokens2 = nltk.word_tokenize(description2)
    stop_words = stopwords.words("english")
    tokens1 = [t for t in tokens1 if t not in stop_words and t not in punctuation]
    tokens2 = [t for t in tokens2 if t not in stop_words and t not in punctuation]
    # 将每个词转换为词向量，并求平均得到描述的向量表示
    vector1 = np.mean([model[t] for t in tokens1 if t in model], axis=0)
    vector2 = np.mean([model[t] for t in tokens2 if t in model], axis=0)
    # 计算余弦相似度，并转换为相似度
    cosine_distance = cosine(vector1, vector2)
    cosine_similarity = 1 - cosine_distance
    # 返回相似度
    return cosine_similarity


# 计算keywords和API之间的相似度，返回一个浮点数
def get_keyword_similarity(keywords, api_title):
    # 这个函数可以根据你的具体相似度度量方法来修改，这里只是一个简单的例子# 假设两个API之间的相似度由它们的描述之间的语义相似度决定，即余弦相似度
    from scipy.spatial.distance import cosine
    from gensim.models import Word2Vec
    # 加载一个预训练的词向量模型，这里使用Google News的模型，你也可以使用其他的模型
    # model = Word2Vec.load("GoogleNews-vectors-negative300.bin")
    # from gensim.models import KeyedVectors  
    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 

    # 获取两个API的描述
    description1 = keywords
    description2 = api_dataset[api_title]["description"]
    # 将描述分词，并过滤掉停用词和标点符号

    from nltk.corpus import stopwords
    from string import punctuation
    nltk.download("punkt")
    nltk.download("stopwords")
    tokens1 = nltk.word_tokenize(description1)
    tokens2 = nltk.word_tokenize(description2)
    stop_words = stopwords.words("english")
    tokens1 = [t for t in tokens1 if t not in stop_words and t not in punctuation]
    tokens2 = [t for t in tokens2 if t not in stop_words and t not in punctuation]
    # 将每个词转换为词向量，并求平均得到描述的向量表示
    vector1 = np.mean([model[t] for t in tokens1 if t in model], axis=0)
    vector2 = np.mean([model[t] for t in tokens2 if t in model], axis=0)
    # 计算余弦相似度，并转换为相似度
    cosine_distance = cosine(vector1, vector2)
    cosine_similarity = 1 - cosine_distance
    # 返回相似度
    return cosine_similarity


# 计算一个关键词在一个Mashup中的权重，返回一个浮点数
def get_keyword_weight(k, m):
    # 这个函数可以根据你的具体权重计算方法来修改，这里只是一个简单的例子# 假设一个关键词在一个Mashup中的权重由它在该Mashup的描述中出现的次数决定，即词频
    from collections import Counter
    # 获取Mashup的描述
    description = mashup_dataset[m]["description"]
    # 将描述分词，并转换为小写import nltk
    nltk.download("punkt")
    tokens = nltk.word_tokenize(description)
    tokens = [t.lower() for t in tokens]
    # 统计每个词出现的次数，并获取关键词的次数
    counter = Counter(tokens)
    frequency = counter[k.lower()]
    # 返回词频作为权重
    return frequency


# 计算一个关键词在一个API中的权重，返回一个浮点数
def get_keyword_api_weight(k, m):
    # 这个函数可以根据你的具体权重计算方法来修改，这里只是一个简单的例子# 假设一个关键词在一个Mashup中的权重由它在该Mashup的描述中出现的次数决定，即词频
    from collections import Counter
    # 获取Mashup的描述
    description = api_dataset[m]["description"]
    # 将描述分词，并转换为小写import nltk
    nltk.download("punkt")
    tokens = nltk.word_tokenize(description)
    tokens = [t.lower() for t in tokens]
    # 统计每个词出现的次数，并获取关键词的次数
    counter = Counter(tokens)
    frequency = counter[k.lower()]
    # 返回词频作为权重
    return frequency


# 在一个图中寻找包含所有关键词节点的最小Steiner树，返回一个子图
# def get_steiner_tree(graph, keywords):
#     # 这个函数可以根据你的具体Steiner树算法来修改，这里只是一个简单的例子# 假设使用一种启发式的算法，即Prim-Dijkstra算法，参考https://en.wikipedia.org/wiki/Steiner_tree_problem#Approximation_algorithmsfrom scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
#     # 创建一个空的子图
#     steiner_tree = nx.Graph()
#     # 将所有关键词节点加入到子图中
#     for k in keywords:
#         steiner_tree.add_node(k)
#     # 对每一对关键词节点，计算它们在原图中的最短路径，并将该路径加入到子图中
#     for i in range(len(keywords)):
#         for j in range(i + 1, len(keywords)):
#             ki = keywords[i]
#             kj = keywords[j]
#             # 使用Dijkstra算法计算最短路径
#             path = nx.dijkstra_path(graph, ki, kj)
#             # 将路径上的所有节点和边加入到子图中
#             steiner_tree.add_nodes_from(path)
#             steiner_tree.add_edges_from(zip(path[:-1], path[1:]))
#     # 返回子图作为Steiner树
#     return steiner_tree

import networkx as nx


def get_steiner_tree(mashup_graph, mashups, query_keywords):
    # 创建一个空的子图  
    steiner_tree = nx.Graph()

    # 将与查询关键词相关的Mashup节点加入到子图中  
    for mashup in mashups:
        for keyword, weight in query_keywords:
            if keyword.lower() in mashups[mashup]["description"].lower():
                steiner_tree.add_node(mashup, weight=weight)

                # 对每一对关键词节点，计算它们在原图中的最短路径，并将该路径加入到子图中
    for i in range(len(query_keywords)):
        for j in range(i + 1, len(query_keywords)):
            ki, _ = query_keywords[i]
            kj, _ = query_keywords[j]
            # 使用Dijkstra算法计算最短路径  
            path = nx.dijkstra_path(mashup_graph, ki, kj)
            # 将路径上的所有节点和边加入到子图中  
            steiner_tree.add_nodes_from(path)
            steiner_tree.add_edges_from(zip(path[:-1], path[1:]))

            # 返回子图作为Steiner树
    return steiner_tree


# 计算两个推荐结果之间的Simhash分数，返回一个整数
def get_simhash_score(r1, r2):
    # 这个函数可以根据你的具体Simhash算法来修改，这里只是一个简单的例子# 假设使用一个32位的Simhash，参考https://en.wikipedia.org/wiki/SimHashfrom simhash import Simhash
    # 将两个推荐结果转换为字符串，用空格分隔每个API的ID
    s1 = " ".join(r1)
    s2 = " ".join(r2)
    # 计算两个字符串的Simhash值
    h1 = simhash.Simhash(s1)
    h2 = simhash.Simhash(s2)
    # 计算两个Simhash值之间的汉明距离，并转换为相似度分数
    hamming_distance = h1.distance(h2)
    simhash_score = 32 - hamming_distance
    # 返回相似度分数
    return simhash_score


# 定义主要的函数
def di_rar(query_input):
    # 输入query
    # query_input = "business"
    # query_input = input("请输入你的查询关键词，用空格分隔：")

    # 实现DI-RAR算法，返回一个API组列表
    # 获取查询关键词
    query_keywords = get_query_keywords(query_input)

    # 获取Mashup数据集和API数据集
    global mashup_dataset, api_dataset
    mashup_path = "active_mashups_data_test2.txt"
    apidata_path = "active_apis_data_test2.txt"
    mashup_dataset = get_mashup_dataset(mashup_path)
    api_dataset = get_api_dataset(apidata_path)

    # 构建加权Mashup相关图
    mashup_graph = nx.Graph()
    for m1 in mashup_dataset:
        for m2 in mashup_dataset:
            if m1 != m2:
                sim = get_mashup_similarity(m1, m2)
                if sim > 0:
                    mashup_graph.add_edge(m1, m2, weight=sim)

    # 对查询关键词进行加权
    keyword_weights = {}
    for k in query_keywords:
        weight = 0
        for m in mashup_dataset:
            weight += get_keyword_weight(k, m)
        keyword_weights[k] = weight

    # 对查询关键词进行排序，取前NUM_KEYWORDS个作为核心需求
    sorted_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)[:NUM_KEYWORDS]

    # 在加权Mashup相关图上执行Steiner树进化检索，得到一个子图
    steiner_tree = get_steiner_tree(mashup_graph, mashup_dataset, sorted_keywords)

    # 从子图中提取出所有使用过的API，作为候选API集合
    candidate_apis = set()
    for m in steiner_tree.nodes():
        candidate_apis.update(mashup_dataset[m]["apis"])

    # 对候选API集合进行聚类，得到若干个API组
    api_clusters = []
    # 这里可以使用任意一种聚类算法，例如K-Means，这里为了简单起见，使用一个贪心的算法
    while candidate_apis:
        # 从候选API集合中随机选择一个API作为初始中心
        center = candidate_apis.pop()
        cluster = [center]
        # 计算其他API与中心的相似度，如果大于某个阈值，则加入该簇
        for api in candidate_apis.copy():
            sim = get_api_similarity(center, api)
            if sim > 0.5:
                cluster.append(api)
                candidate_apis.remove(api)
        # 将该簇加入到API组列表中
        api_clusters.append(cluster)

    # 对每个API组进行评分，根据核心需求和非核心需求的匹配程度
    api_scores = {}
    for cluster in api_clusters:
        score = 0
        # 计算核心需求的匹配程度，即每个核心关键词在该簇中的最大权重之和
        for k, w in sorted_keywords:
            max_weight = 0
            for api in cluster:
                weight = get_keyword_api_weight(k, api)
                if weight > max_weight:
                    max_weight = weight
            score += w * max_weight
        # 计算非核心需求的匹配程度，即该簇中所有API的描述与查询关键词的相似度之和
        for api in cluster:
            for k in query_keywords:
                sim = get_keyword_similarity(k, api)
                score += sim
        # 将该簇的评分存入字典中
        api_scores[tuple(cluster)] = score

    # 对API组进行排序，取前NUM_RESULTS个作为推荐结果
    sorted_results = sorted(api_scores.items(), key=lambda x: x[1], reverse=True)[:NUM_RESULTS]

    # 使用Simhash技术对推荐结果进行去重，过滤掉相似度高于阈值的结果
    final_results = []
    for r, s in sorted_results:
        is_duplicate = False
        for f in final_results:
            score = get_simhash_score(r, f)
            if score < THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            final_results.append(r)

    # 返回最终的推荐结果列表
    return final_results


final_results = di_rar("business")
print(final_results)

# {
#     "query":"business"
# }
# @app.route('/api/di_rar', methods=['POST'])
# def api():
#     # 从请求中获取 JSON 数据
#     data = request.get_json()
#
#     # 从 JSON 数据中提取两个数字
#     input = data.get('query')
#     final_results = di_rar(input)
#
#     # 返回带有结果的 JSON 响应
#     return jsonify({'result': final_results})

# if __name__ == '__main__':
#     app.run(debug=True)
