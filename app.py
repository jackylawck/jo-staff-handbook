import streamlit as st
import tempfile
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

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
    .source-tag {
        font-size: 0.8em;
        color: #198754;
        font-weight: bold;
        margin-bottom: 5px;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 🌳 確定性目錄索引與權重路由
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

def analyze_intent_and_route(query, last_chapters=None):
    q_lower = query.lower()
    scores = {}
    
    for kw, ch_id in MANUAL_INDEX_TREE["keywords_to_chapter"].items():
        if kw in q_lower:
            scores[ch_id] = scores.get(ch_id, 0) + 2
            
    for ch in MANUAL_INDEX_TREE["chapters"]:
        for intent in ch["intents"]:
            if intent.lower() in q_lower:
                scores[ch["id"]] = scores.get(ch["id"], 0) + 1
                
    if not scores and last_chapters:
        return last_chapters
        
    sorted_chs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    result = []
    for ch_id, _ in sorted_chs:
        for ch in MANUAL_INDEX_TREE["chapters"]:
            if ch["id"] == ch_id:
                result.append(ch["title"])
                
    return result

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ==========================================
# 3. 🧠 確定性頁碼映射與解析引擎
# ==========================================
PAGE_CHAPTER_MAP = {
    2: "序言", 3: "遠景及使命", 4: "目錄", 
    8: "第一章 公司簡介", 10: "第二章 員工手冊簡介", 11: "第三章 僱傭條款",
    15: "第四章 辦公時間及考勤", 18: "第五章 薪酬管理", 20: "第六章 假期",
    28: "第七章 員工福利及保障", 30: "第八章 績效考核", 31: "第九章 培訓和發展",
    32: "第十章 紀律守則及防貪誠信守則", 43: "第十一章 員工關係及溝通",
    44: "第十二章 附則", 45: "附則一：保密協議", 47: "附則二：員工使用電子通訊系統指引",
    49: "員工手冊確認書"
}

def get_chapter_by_page(doc_page_num):
    actual_page = doc_page_num + 1
    current_ch = "通用條文"
    for p in sorted(PAGE_CHAPTER_MAP.keys()):
        if actual_page >= p:
            current_ch = PAGE_CHAPTER_MAP[p]
        else:
            break
    return current_ch

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
        
        raw_chunks = text_splitter.split_documents(documents)
        
        for chunk in raw_chunks:
            page_num = chunk.metadata.get("page", 0)
            chapter_title = get_chapter_by_page(page_num)
            
            chunk.metadata["chapter"] = chapter_title
            chunk.page_content = f"[{chapter_title}]\n{chunk.page_content}"
            chunks.append(chunk)
            
    except Exception as e:
        st.error(f"解析 PDF 檔案時出錯: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
    return chunks

# ==========================================
# 4. 主畫面佈局與狀態管理
# ==========================================
st.title("🏗️ 東淦工程有限公司 (Jumbo Orient)")
st.subheader("東淦員工手冊智能查詢系統 (jo-staff)")

st.info(
    "🔒 **內部數據安全保障：**\n"
    "本系統採用純本地數據比對技術。為保障私隱，當您點擊「清除對話」或關閉網頁時，所有紀錄將被徹底銷毀。"
)

if 'jo_messages' not in st.session_state:
    st.session_state.jo_messages = []
if 'last_chapters' not in st.session_state:
    st.session_state.last_chapters = []

ensemble_retriever = None
all_chunks = []
uploaded_file_names = []

with st.sidebar:
    st.header("📂 員工手冊上傳")
    uploaded_files = st.file_uploader(
        "請上傳最新版《月薪員工手冊》PDF 檔案", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("🗑️ 清除對話與數據", use_container_width=True):
        st.session_state.jo_messages = []
        st.session_state.last_chapters = []
        st.rerun()

if uploaded_files:
    for f in uploaded_files:
        all_chunks.extend(process_pdf_to_chunks(f))
        uploaded_file_names.append(f.name)
    
    if all_chunks:
        with st.spinner('構建雙引擎混合檢索矩陣中 (FAISS + BM25)...'):
            # 建立語意向量檢索器 (FAISS)
            embeddings = get_embedding_model()
            vector_db = FAISS.from_documents(all_chunks, embeddings)
            faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 4})
            
            # 建立關鍵字檢索器 (BM25)
            bm25_retriever = BM25Retriever.from_documents(all_chunks)
            bm25_retriever.k = 4
            
            # 組合雙引擎 (權重：語意 60% / 關鍵字 40%)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], 
                weights=[0.4, 0.6]
            )

with st.sidebar:
    st.markdown("---")
    st.header("📊 知識庫加載狀態")
    if uploaded_files:
        st.write(f"📁 已加載檔案數：{len(uploaded_files)} 份")
        st.write(f"🧩 結構化解析段落：{len(all_chunks)} 段")
        st.success("✅ 混合檢索雙引擎已啟動")
    else:
        st.write("尚未上傳檔案。")
    st.caption("🌐 **公司網站：** [jumboorient.com.hk](https://jumboorient.com.hk/)")

# ==========================================
# 5. 快捷提問區
# ==========================================
prompt = st.chat_input("請輸入您關於人事政策的疑問...")

if not prompt and (ensemble_retriever is not None):
    st.markdown("<div class='quick-btn-container'><b>💡 常見問題快速查詢：</b></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🌪️ 颱風黑雨安排", use_container_width=True): prompt = "八號風球或者黑雨，洗唔洗返工？"
    if col2.button("🤒 病假申請流程", use_container_width=True): prompt = "請病假要點樣申請？有幾多日有薪病假？"
    if col3.button("💼 辭職通知期", use_container_width=True): prompt = "辭職要補幾多日通知期？"
    if col4.button("🧧 年終花紅雙糧", use_container_width=True): prompt = "年終雙糧同花紅點樣計算？"

# ==========================================
# 6. 智能對話與混合檢索
# ==========================================
for msg in st.session_state.jo_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt:
    st.session_state.jo_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if ensemble_retriever is None or not all_chunks:
            error_msg = "🛑 **系統提示：** 請先在左側上傳《員工手冊》PDF 檔案。"
            st.error(error_msg)
            st.session_state.jo_messages.append({"role": "assistant", "content": error_msg})
        else:
            routed_chapters = analyze_intent_and_route(prompt, st.session_state.last_chapters)
            st.session_state.last_chapters = routed_chapters
            
            enhanced_prompt = prompt
            routing_notice = ""
            if routed_chapters:
                chapters_str = "、".join(routed_chapters)
                enhanced_prompt = f"{prompt} 相關章節聚焦：{chapters_str}"
                routing_notice = f"<div class='routing-notice'>🔍 系統判定查詢意圖，已自動聚焦於：<b>{chapters_str}</b></div>"
                st.markdown(routing_notice, unsafe_allow_html=True)

            # 執行混合檢索 (Ensemble Retriever 會自動處理 RRF 排序)
            retrieved_docs = ensemble_retriever.invoke(enhanced_prompt)
            
            if retrieved_docs:
                combined_content = ""
                seen_content_signatures = set()
                
                # 取 Top 3 最佳結果顯示
                for doc in retrieved_docs[:3]:
                    # 優化去重指紋：只取前 50 個乾淨字元，解決微小差異造成的誤判
                    clean_signature = re.sub(r'\W+', '', doc.page_content)[:50]
                    if clean_signature not in seen_content_signatures:
                        seen_content_signatures.add(clean_signature)
                        
                        chapter_tag = doc.metadata.get("chapter", "通用條文")
                        clean_content = doc.page_content.replace(f"[{chapter_tag}]\n", "").replace("\n", "<br>")
                        combined_content += f"<span class='source-tag'>📍 來源節錄：{chapter_tag}</span>{clean_content}<br><hr><br>"

                st.markdown(f"<div class='confidence-badge'>✅ 綜合檢索結果 (語意 + 關鍵字)</div>", unsafe_allow_html=True)
                
                source_files = ", ".join(set(uploaded_file_names))
                response_html = (
                    f"{routing_notice}"
                    f"<div class='answer-box'>"
                    f"<b>📋 擷取原文（來源：{source_files}）：</b><br><br>{combined_content}"
                    f"</div>"
                    f"<small><i>※ 提示：此為系統根據手冊自動比對的結果，若需處理具體個案，請聯絡人力資源組。</i></small>"
                )
                
                st.markdown(response_html.replace(routing_notice, ""), unsafe_allow_html=True) 
                st.session_state.jo_messages.append({"role": "assistant", "content": response_html})
            else:
                fallback_msg = "抱歉，在手冊中找不到高度相關的條文。建議您換個說法，或聯絡人力資源組。"
                st.warning(fallback_msg)
                st.session_state.jo_messages.append({"role": "assistant", "content": fallback_msg})
