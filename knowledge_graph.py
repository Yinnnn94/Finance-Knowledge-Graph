import re
import jieba
import spacy
def clean_text(text):
    # 移除標點符號
    text = re.sub(r'[^\w\s]', '', text)
    # 移除換行符號
    return text.split()

# 讀取資料
with open("cube_introduction.txt", "r") as file:
    text = file.read() # 一行一行讀取
    # print(','.join(jieba.cut(text)))
text = clean_text(text) # 清理資料
sentence = [','.join(jieba.cut(t, cut_all=False)) for t in text]
print(sentence)
