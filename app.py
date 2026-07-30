import streamlit as st
import tempfile
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==========================================
# 1. 頁面配置與 UI 樣式
# ==========================================
st.set_page_config(
    page_title="東淦員工手冊智能查詢系統",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .confidence-badge {
        background-color: #0d6efd !important;
        color: #ffffff !important;
        padding: 6px 12px !important;
        border-radius: 20px !important;
        font-size: 0.85em !important;
        font-weight: bold !important;
        display: inline-block !important;
        margin-bottom: 10px;
    }
    .answer-box {
        background-color: #f8f9fa;
        border-left: 5px solid #0d6efd;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
        line-height: 1.6;
        color: #333333;
    }
    .routing-notice {
        color: #6c757d;
        font-size: 0.9em;
        margin-bottom: 10px;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 🌳 目錄索引樹 (Directory Index Tree)
# 僅包含結構化元數據，絕不包含具體條文。對應手冊章節框架。
# ==========================================
MANUAL_INDEX_TREE = {
    "chapters": [
        {"id": "ch3", "title": "第三章 僱傭條款", "intents": ["試用期", "調職", "離職", "終止合約", "解僱", "退休", "證明書"]},
        {"id": "ch4", "title": "第四章 辦公時間及考勤", "intents": ["上班時間", "考勤", "遲到", "早退", "拍卡", "惡劣天氣", "打風", "黑雨", "颱風"]},
        {"id": "ch5", "title": "第五章 薪酬管理", "intents": ["工資", "底薪", "出糧", "雙糧", "花紅", "加薪", "調薪"]},
        {"id": "ch6", "title": "第六章 假期", "intents": ["休息日", "公眾假期", "年假", "補假", "病假", "事假", "產假", "分娩假", "侍產假", "婚假", "恩恤假", "生日假", "無薪假"]},
        {"id": "ch7", "title": "第七章 員工福利及保障", "intents": ["強積金", "MPF", "醫療", "門診", "住院", "睇醫生", "康樂"]},
        {"id": "ch8", "title": "第八章 績效考核", "intents": ["考核", "評估", "appraisal", "最佳員工", "獎懲"]},
        {"id": "ch9", "title": "第九章 培訓和發展", "intents": ["培訓", "進修", "資助", "課程"]},
        {"id": "ch10", "title": "第十章 紀律守則及防貪誠信守則", "intents": ["紀律", "行為", "防貪", "利益衝突", "賄賂", "收禮", "處分", "警告"]},
        {"id": "appx", "title": "附則", "intents": ["保密協議", "電子通訊", "電腦使用"]}
    ],
    "keywords_to_chapter": {
        "辭職": "ch3", "通知期": "ch3", "試用": "ch3", 
        "打風": "ch4", "黑雨": "ch4", "ot": "ch4", "拍卡": "ch4",
        "雙糧": "ch5", "花紅": "ch5", "報稅": "ch5",
        "大假": "ch6", "al": "ch6", "sl": "ch6", "醫生紙": "ch6", "補假": "ch6",
        "claim錢": "ch7", "洗牙": "ch7", "津貼": "ch7",
        "升職": "ch8", "上堂": "ch9", 
        "請客": "ch10", "利是": "ch10", "賭錢": "ch10", "保密": "appx"
    }
}

def analyze_intent_and_route(query):
    """
    意圖解析與路由：根據提問回傳對應的章節標題，用於引導檢索。
    """
    q_lower = query.lower()
    target_chapters = set()
    
    # 1. 精確關鍵字映射
    for kw, ch_id in MANUAL_INDEX_TREE["keywords_to_chapter"].items():
        if kw in q_lower:
            for ch in MANUAL_INDEX_TREE["chapters"]:
                if ch["id"] == ch_id:
                    target_chapters.add(ch["title"])
    
    # 2. 意圖標籤模糊匹配
    for ch in MANUAL_INDEX_TREE["chapters"]:
        for intent in ch["intents"]:
            if intent.lower() in q_lower:
                target_chapters.add(ch["title"])
                
    return list(target_chapters)[:2] # 最多回傳 2 個最相關章節

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ==========================================
# 3. 🧠 PDF 結構化解析與文本前綴注入 (Context Injection)
# ==========================================
def process_pdf_to_chunks(uploaded_file):
    chunks = []
    tmp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # 使用正則表達式在切塊前嘗試標記章節
        current_chapter_title = "通用條文"
        chapter_pattern = re.compile(r'(第[一二三四五六七八九十]+章.*?|附則[一二].*?)\n')
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n第", "\n\n", "\n", "。", " "]
        )
        
        raw_chunks = text_splitter.split_documents(documents)
        
        # 將章節上下文注入到每個 chunk 中
        for chunk in raw_chunks:
            match = chapter_pattern.search(chunk.page_content)
            if match:
                current_chapter_title = match.group(1).strip()
            
            # 文本前綴注入：確保模型知道這段話屬於哪個結構
            chunk.page_content = f"[{current_chapter_title}]\n{chunk.page_content}"
            chunks.append(chunk)
            
    except Exception as e:
        st.error(f"解析 PDF 檔案時出錯: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
    return chunks

# ==========================================
# 4. 主畫面佈局
# ==========================================
st.title("🏗️ 東淦工程有限公司 (Jumbo Orient)")
st.subheader("東淦員工手冊智能查詢系統 (jo-staff)")

st.info(
    "🔒 **內部數據安全保障：**\n"
    "本系統採用純本地數據比對技術。當您點擊「清除對話」或關閉網頁時，所有紀錄將被徹底銷毀。"
)

vector_db = None
all_chunks = []
uploaded_file_names = []

with st.sidebar:
    st.header("📂 員工手冊上傳")
    uploaded_files = st.file_uploader(
        "請上傳公司《月薪員工手冊》PDF 檔案", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("🗑️ 清除對話與數據", use_container_width=True):
        st.session_state.jo_messages = []
        st.rerun()

if uploaded_files:
    for f in uploaded_files:
        all_chunks.extend(process_pdf_to_chunks(f))
        uploaded_file_names.append(f.name)
    if all_chunks:
        with st.spinner('構建章節路由與智能知識庫中，請稍候...'):
            embeddings = get_embedding_model()
            vector_db = FAISS.from_documents(all_chunks, embeddings)

with st.sidebar:
    st.markdown("---")
    st.header("📊 知識庫加載狀態")
    st.write(f"📁 已加載檔案數：{len(uploaded_files) if uploaded_files else 0} 份")
    st.write(f"🧩 結構化解析段落：{len(all_chunks)} 段")
    st.caption("🌐 **公司網站：** [jumboorient.com.hk](https://jumboorient.com.hk/)")

# ==========================================
# 5. 快捷提問區
# ==========================================
prompt = st.chat_input("請輸入您關於人事政策的疑問...")

if not prompt and (vector_db is not None):
    st.markdown("<div class='quick-btn-container'><b>💡 常見問題快速查詢：</b></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🌪️ 颱風黑雨安排", use_container_width=True): prompt = "八號風球或者黑雨，洗唔洗返工？"
    if col2.button("🤒 病假申請流程", use_container_width=True): prompt = "請病假要點樣申請？有幾多日有薪病假？"
    if col3.button("💼 辭職通知期", use_container_width=True): prompt = "辭職要補幾多日通知期？"
    if col4.button("🧧 年終花紅雙糧", use_container_width=True): prompt = "年終雙糧同花紅點樣計算？"

# ==========================================
# 6. 智能對話與兩階段檢索
# ==========================================
if 'jo_messages' not in st.session_state:
    st.session_state.jo_messages = []

for msg in st.session_state.jo_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt:
    st.session_state.jo_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db is None or not all_chunks:
            error_msg = "🛑 **系統提示：** 請先在左側上傳《員工手冊》PDF 檔案。"
            st.error(error_msg)
            st.session_state.jo_messages.append({"role": "assistant", "content": error_msg})
        else:
            # 階段一：意圖解析與路由
            routed_chapters = analyze_intent_and_route(prompt)
            
            # 構建增強搜尋字串 (將目標章節隱含加入，限制向量庫的注意力)
            enhanced_prompt = prompt
            routing_notice = ""
            if routed_chapters:
                chapters_str = "、".join(routed_chapters)
                enhanced_prompt = f"{prompt} 相關章節聚焦：{chapters_str}"
                routing_notice = f"<div class='routing-notice'>🔍 系統判定查詢意圖，已自動將檢索範圍聚焦於：<b>{chapters_str}</b></div>"
                st.markdown(routing_notice, unsafe_allow_html=True)

            # 階段二：向量精確檢索
            docs_and_scores = vector_db.similarity_search_with_score(enhanced_prompt, k=3)
            
            if docs_and_scores:
                top_score = docs_and_scores[0][1]
                
                if top_score < 0.8: relevance_label = "🌟 高度相關"
                elif top_score < 1.2: relevance_label = "✅ 具參考價值"
                else: relevance_label = "⚠️ 關聯性較低 (請確認關鍵字)"

                combined_content = "<br><hr><br>".join([doc.page_content.replace("\n", "<br>") for doc, score in docs_and_scores])
                source_files = ", ".join(set(uploaded_file_names))

                st.markdown(f"<div class='confidence-badge'>{relevance_label}</div>", unsafe_allow_html=True)
                
                response_html = (
                    f"{routing_notice}"
                    f"<div class='answer-box'>"
                    f"<b>📋 擷取原文（來源：{source_files}）：</b><br><br>{combined_content}"
                    f"</div>"
                    f"<small><i>※ 提示：此為系統初步比對結果，若有疑慮請聯絡人力資源組。</i></small>"
                )
                
                st.markdown(response_html.replace(routing_notice, ""), unsafe_allow_html=True) 
                st.session_state.jo_messages.append({"role": "assistant", "content": response_html})
            else:
                fallback_msg = "抱歉，在手冊中找不到高度相關的條文。建議您換個說法，或聯絡人力資源組。"
                st.warning(fallback_msg)
                st.session_state.jo_messages.append({"role": "assistant", "content": fallback_msg})
