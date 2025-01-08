import re
import jieba
import spacy
from rank_bm25 import BM25Okapi
from transformers import pipeline
from transformers import  AutoTokenizer, AutoModelForQuestionAnswering
import gradio as gr

def clean_text(text):
    # 移除標點符號
    text = re.sub(r'[^\w\s]', '', text)
    # 移除換行符號
    return text
    
def prep_source():
    paragraph = []
    # 讀取資料
    with open("cube_introduction.txt", "r") as file:
        text = file.read() # 一行一行讀取

        text = text.replace("\n", ',') # 以換行符號分割
        para = text.split("---") # --- 分段

    for u in para:
        paragraph.append(u)
    return paragraph



def bm25_search(query):
    paragraph = prep_source()
    bm25 = BM25Okapi(paragraph) 
    """使用 BM25 搜索最相關的段落"""
    topk_idx = bm25.get_top_n(query, paragraph, n=1)  # 取得最相關的段落
    source_text = ' '.join(topk_idx)  # 將列表轉為字串
    return source_text

def text_generator(query):
    model_name = 'NchuNLP/Chinese-Question-Answering'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    QA = pipeline("question-answering", model=model, tokenizer=tokenizer)
    source = bm25_search(str(query))  # 使用 BM25 搜索相關段落
    result = QA(question = query, context = source)  # 問答模型生成答案
    return result['answer']

# Q&A

if __name__ == "__main__":
    demo = gr.Blocks()
    with demo:
        gr.Markdown("## CUBE信用卡問答")
        with gr.Row():
            text_input = gr.Textbox(label="輸入問題")
            text_output = gr.Textbox(label="答案")
        buttom = gr.Button("送出")
        buttom.click(text_generator, inputs=text_input, outputs=text_output)  # 傳入函數本身
            
        demo.launch()


# query = ["CUBE卡綁定LINE Pay回饋多少？", "首刷好禮為何？", "刷星巴克最高回饋幾％？", "慶生月外送回饋多少？"]
# for i in query:
#     print(f"""Query: {i}\nresponse: {text_generator(i)}""")
#     print('-' * 10)

