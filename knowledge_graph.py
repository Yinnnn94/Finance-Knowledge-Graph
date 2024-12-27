import re
import jieba
import spacy
from rank_bm25 import BM25Okapi

def clean_text(text):
    # 移除標點符號
    text = re.sub(r'[^\w\s]', '', text)
    # 移除換行符號
    return text
paragraph = []
# 讀取資料
with open("cube_introduction.txt", "r") as file:
    text = file.read() # 一行一行讀取
    text = text.replace("\n", ',') # 以換行符號分割
    para = text.split("---") # --- 分段

for u in para:
    text = clean_text(u) # 清理資料
    paragraph.append(text)

# Q&A
bm25 = BM25Okapi(paragraph)
query = "CUBE卡綁定LINE Pay回饋多少？"
# tokenized_query = ' '.join(jieba.cut(query, cut_all=False))
doc_scores = bm25.get_scores(query)
topk = 1
topk_idx = bm25.get_top_n(query, paragraph, n=topk)
print(f"Query: {query}")
print(topk_idx, doc_scores[0]) 


