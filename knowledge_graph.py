import re
import jieba
import spacy
from rank_bm25 import BM25Okapi
from transformers import pipeline


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
    paragraph.append(u)

def text_generator(query, source):
    text2text = pipeline("text2text-generation", model="t5-small", device = 0)
    response = text2text(f'用戶問{query}, 基於此回答{source}', max_length=50)
    return response[0]['generated_text']
# Q&A
bm25 = BM25Okapi(paragraph)
query = ["CUBE卡綁定LINE Pay回饋多少？", "活動期限為何？", "哪些餐廳有8%回饋？"]
# tokenized_query = ' '.join(jieba.cut(query, cut_all=False))
for i in query:
    doc_scores = bm25.get_scores(i)
    topk = 1
    topk_idx = bm25.get_top_n(i, paragraph, n=topk)
    print('-' * 100)
    print(f"Query: {i}")
    print(text_generator(i, topk_idx))
