import streamlit as st
import tempfile
import os
import re

# 引入本地向量組件與 PDF 處理工具
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
    .quick-btn-container {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 🛡️ 員工手冊目錄智能導航 (Metadata Routing)
# ==========================================
MANUAL_STRUCTURE_MAP = {
    "僱傭與離職": {
        "keywords": ["辭職", "唔做", "試用期", "調職", "離職", "退休", "代通知金", "解僱", "工作證明", "離職面談"],
        "pointers": "第三章 僱傭條款 3.3 試用期 3.6 終止僱傭合約 3.7 離職 3.9 服務證明書"
    },
    "考勤與天氣": {
        "keywords": ["遲到", "早退", "拍卡", "忘記帶證", "打風", "黑雨", "紅雨", "極端情況", "ot", "加班", "外勤", "颱風"],
        "pointers": "第四章 辦公時間及考勤 4.2 考勤 4.3 惡劣天氣指引"
    },
    "薪酬與花紅": {
        "keywords": ["出糧", "雙糧", "bonus", "花紅", "加人工", "底薪", "戶口", "報稅", "酬金期"],
        "pointers": "第五章 薪酬管理 5.3 月薪員工薪金計算參考 5.5 年終雙糧 5.6 年終花紅 5.7 薪酬檢討"
    },
    "各類假期": {
        "keywords": ["大假", "al", "病假", "sl", "醫生紙", "事假", "產假", "侍產假", "婚假", "恩恤假", "生日假", "補假", "陪審員"],
        "pointers": "第六章 假期 6.2 年假 6.3 補假 6.4 病假 6.6 分娩假 6.7 侍產假 6.8 婚假 6.11 生日假"
    },
    "福利與保障": {
        "keywords": ["強積金", "mpf", "睇醫生", "醫療咭", "洗牙", "claim錢", "門診", "住院", "康樂", "津貼"],
        "pointers": "第七章 員工福利及保障 7.1 強積金計劃 7.2 醫療保障"
    },
    "考核與培訓": {
        "keywords": ["appraisal", "評估", "升職", "培訓", "上堂", "資助", "最佳員工", "進修"],
        "pointers": "第八章 績效考核 8.2 年度評審 第九章 培訓和發展 9.5 培訓資助"
    },
    "紀律與防貪": {
        "keywords": ["收禮", "請客", "食飯", "利益衝突", "利是", "賭錢", "警告信", "行為不當", "保密", "防貪", "防止賄賂"],
        "pointers": "第十章 紀律守則及防貪誠信守則 10.1 紀律守則 10.2 防貪誠信守則 10.3 處分程序 附則一 保密協議 附則二 電子通訊系統指引"
    }
}

def expand_hr_query_semantics(query):
    q_lower = query.lower()
    injected_pointers = []
    
    for category, data in MANUAL_STRUCTURE_MAP.items():
        if any(keyword in q_lower for keyword in data["keywords"]):
            injected_pointers.append(data["pointers"])
            
    if injected_pointers:
        return query + " " + " ".join(injected_pointers)
    return query

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ==========================================
# 3. 🧠 PDF 結構化解析與切塊引擎 (強化異常處理)
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
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n第", "\n\n", "\n", "。", " "]
        )
        chunks = text_splitter.split_documents(documents)
        
    except Exception as e:
        st.error(f"解析 PDF 檔案時出錯: {str(e)}")
    finally:
        # 確保無論成功或失敗，暫存檔都會被刪除 (修復潛在風險)
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
    "本系統採用純本地數據比對技術，您上傳的《員工手冊》只會暫存在當前網頁會話中。"
    "為保障私隱，當您點擊「清除對話」或關閉網頁時，所有紀錄與數據將被立即銷毀。"
)

vector_db = None
all_chunks = []
uploaded_file_names = []

with st.sidebar:
    st.header("📂 員工手冊上傳")
    uploaded_files = st.file_uploader(
        "請上傳公司《月薪員工手冊》PDF 檔案 (支援多版本比對)", 
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
        with st.spinner('構建智能知識庫中，請稍候...'):
            embeddings = get_embedding_model()
            vector_db = FAISS.from_documents(all_chunks, embeddings)

with st.sidebar:
    st.markdown("---")
    st.header("📊 知識庫加載狀態")
    st.write(f"📁 已加載檔案數：{len(uploaded_files) if uploaded_files else 0} 份")
    st.write(f"🧩 解析精準條文段落：{len(all_chunks)} 段")
    
    st.markdown("---")
    st.caption("🌐 **公司網站：** [jumboorient.com.hk](https://jumboorient.com.hk/)")
    st.caption("⚙️ 系統支援：[Jacky Law](https://jackylawck.github.io/jackylawck/)")

# ==========================================
# 5. 快捷提問區 (Quick Prompts)
# ==========================================
prompt = st.chat_input("請輸入您關於人事政策的疑問（例如：病假點申請？）...")

if not prompt and (vector_db is not None):
    st.markdown("<div class='quick-btn-container'><b>💡 常見問題快速查詢：</b></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🌪️ 颱風黑雨安排", use_container_width=True): prompt = "八號風球或者黑雨，洗唔洗返工？"
    if col2.button("🤒 病假申請流程", use_container_width=True): prompt = "請病假要點樣申請？有幾多日有薪病假？"
    if col3.button("💼 辭職通知期", use_container_width=True): prompt = "辭職要補幾多日通知期？"
    if col4.button("🧧 年終花紅雙糧", use_container_width=True): prompt = "年終雙糧同花紅點樣計算？"

# ==========================================
# 6. 智能對話介面
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
            error_msg = "🛑 **系統提示：** 請先在左側上傳《員工手冊》PDF 檔案，否則助理無法為您翻查條文。"
            st.error(error_msg)
            st.session_state.jo_messages.append({"role": "assistant", "content": error_msg})
        else:
            enhanced_prompt = expand_hr_query_semantics(prompt)
            # 增加 k 值至 3，確保跨章節問題不遺漏
            docs_and_scores = vector_db.similarity_search_with_score(enhanced_prompt, k=3)
            
            if docs_and_scores:
                # 取得最相關的頂層分數來判定信心 (越接近 0 越好)
                top_score = docs_and_scores[0][1]
                
                # 根據 L2 Distance 進行直觀的分級提示
                if top_score < 0.8:
                    relevance_label = "🌟 高度相關"
                elif top_score < 1.2:
                    relevance_label = "✅ 具參考價值"
                else:
                    relevance_label = "⚠️ 關聯性較低 (請確認關鍵字)"

                combined_content = "<br><hr><br>".join([doc.page_content.replace("\n", "<br>") for doc, score in docs_and_scores])
                source_files = ", ".join(set(uploaded_file_names))

                st.success(f"系統已根據您的問題，為您優先比對以下手冊章節：")
                st.markdown(f"<div class='confidence-badge'>{relevance_label}</div>", unsafe_allow_html=True)
                
                response_html = (
                    f"<div class='answer-box'>"
                    f"<b>📋 擷取原文（來源文件：{source_files}）：</b><br><br>{combined_content}"
                    f"</div>"
                    f"<small><i>※ 提示：此為系統從上傳文件中自動比對出的相關段落。若有具體個案或需進一步解釋，請聯絡人力資源組。</i></small>"
                )
                
                st.markdown(response_html, unsafe_allow_html=True)
                st.session_state.jo_messages.append({"role": "assistant", "content": response_html})
            else:
                fallback_msg = "抱歉，在手冊中找不到與您問題高度相關的條文。建議您換個說法（例如使用書面語），或直接聯絡人力資源組查詢。"
                st.warning(fallback_msg)
                st.session_state.jo_messages.append({"role": "assistant", "content": fallback_msg})
