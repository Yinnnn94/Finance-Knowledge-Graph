import re
import jieba
import spacy
from rank_bm25 import BM25Okapi
from transformers import pipeline
from transformers import  AutoTokenizer, AutoModelForQuestionAnswering


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
    model_name = 'NchuNLP/Chinese-Question-Answering'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    QA = pipeline("question-answering", model = model, tokenizer = tokenizer)    

    result = QA(question = query, context = source)
    # 輸出生成的答案
    return result['answer']

# Q&A

if __name__ == "__main__":

    bm25 = BM25Okapi(paragraph)
    query = ["CUBE卡綁定LINE Pay回饋多少？", "首刷好禮為何？", "刷星巴克最高回饋幾％？", "慶生月外送回饋多少？"]
    for i in query:
        doc_scores = bm25.get_scores(i)
        topk = 1
        topk_idx = bm25.get_top_n(i, paragraph, n=topk)
        source_text = ' '.join(topk_idx)  # 將列表轉換為字串
        print(f"""Query: {i}\nresponse: {text_generator(i, source_text)}\n""")
        print('-' * 10)

