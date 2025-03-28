# 编辑举例
from rapidfuzz import fuzz

def correct_query_by_edit_distance(query, candidates, threshold=0):
    best_match = None
    max_similarity = 0  # 初始化最大相似度为0
    for candidate in candidates:
        similarity = fuzz.ratio(query, candidate)
        print(f"Similarity between '{query}' and '{candidate}': {similarity}")
        if similarity > threshold:
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = candidate
    return best_match if best_match is not None else query

# 示例
candidates = ["永定河孔雀城瞰璟园", "永定河孔雀城秋月园", "永定河孔雀城大湖"]
query = "永定河孔雀城秋瞰璟"
corrected_query = correct_query_by_edit_distance(query, candidates)
print(f"Corrected Query: {corrected_query}")

exit(0)

# 基于拼音实现纠正
from pypinyin import lazy_pinyin, Style
from collections import Counter

def remove_tones(pinyin):
    """移除拼音中的声调"""
    return ''.join([char for char in pinyin if char.isdigit() == False])

def correct_query_by_pinyin(query, candidates):
    pinyin_query = lazy_pinyin(query, style=Style.NORMAL)
    pinyin_candidates = [lazy_pinyin(candidate, style=Style.NORMAL) for candidate in candidates]

    # print(pinyin_candidates, candidates)
    matching_candidates = []
    for pinyin_candidate, candidate in zip(pinyin_candidates, candidates):
        print(''.join(pinyin_query))
        print(''.join(pinyin_candidate))
        if ''.join(pinyin_query) == ''.join(pinyin_candidate):
            matching_candidates.append(candidate)
    
    if matching_candidates:
        most_common = Counter(matching_candidates).most_common(1)[0][0]
        return most_common
    else:
        return query

# 示例
candidates = ["苹果", "平果", "萍果"]
query = "pingguo"
corrected_query = correct_query_by_pinyin(query, candidates)
print(f"Corrected Query: {corrected_query}")

exit(0)

#基于点击行为的 queryN-docN 协同过滤算法纠正,计算query相似度

import pandas as pd
from surprise import Reader, Dataset, KNNBasic


# 构建点击行为数据
data = {
    'query': ['search', 'serach', 'search', 'search——'],
    'doc': ['doc1', 'doc1', 'doc2', 'doc3'],
    'clicks': [1, 1, 1, 1]
}
df = pd.DataFrame(data)
reader = Reader(rating_scale=(1, 1))
data = Dataset.load_from_df(df[['query', 'doc', 'clicks']], reader)

# 构建相似度矩阵
trainset = data.build_full_trainset()
sipym_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sipym_options)
algo.fit(trainset)

item_similarity_matrix = algo.sim
print(item_similarity_matrix)
# 计算相似度
query_similarities = algo.compute_similarities()

# 输出相似度最高的查询
#query_index = trainset.to_inner_iid('search')
query_index = trainset.to_inner_uid('search')
similar_queries = sorted(enumerate(query_similarities[query_index]), key=lambda x: x[1], reverse=True)
#top_similar_queries = [trainset.to_raw_iid(idx) for idx, sim in similar_queries[:5]]
top_similar_queries = [trainset.to_raw_uid(idx) for idx, sim in similar_queries[:5]]
print(f"Similar Queries: {top_similar_queries}")

exit(0)



# 基于 session embedding 挖掘更多的 query 序列进行纠错
import numpy as np
from gensim.models import Word2Vec

# 构建查询序列
queries = [
    ["search", "weather"],
    ["serach", "weather"],
    ["search_", "news"],
    ["search", "sports"]
]

# 训练 word2vec 模型pipeline
model = Word2Vec(sentences=queries, vector_size=50, window=5, min_count=1, workers=4)

# 假设 model 已经训练完成
# 寻找相似词
similar_words = model.wv.most_similar('serach', topn=5)
print(similar_words)

# 纠错
query = "sreach news"

words = query.split()
for i, word in enumerate(words):
    if word not in model.wv.key_to_index:
        print([w for w in words if w != word])
        # 寻找words中最相似词
        closest_word = model.wv.most_similar(positive=[w for w in words if w != word], topn=2)
        print(closest_word)
        if closest_word:
            words[i] = closest_word[0][0]

corrected_query = ' '.join(words)
print(f"Corrected Query: {corrected_query}")
