import streamlit as st
import tempfile
import os
import re
import json
from datetime import datetime

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
    .override-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
        line-height: 1.6;
        color: #664d03;
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
    .z-drive-path {
        background-color: #e2e3e5;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        color: #383d41;
    }
    .policy-effective {
        font-size: 0.85em;
        color: #856404;
        background-color: #fff3cd;
        padding: 2px 10px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 📂 內建政策 + 外部 JSON 動態載入
# ==========================================
DEFAULT_POLICY_OVERRIDES = [
    {
        "id": "salary_structure_2026",
        "keywords": ["雙糧", "13個月", "年終雙糧", "花紅", "酌情花紅", "薪酬架構"],
        "title": "📌 【最新薪酬架構】2026年1月1日起生效",
        "summary": (
            "• <b>制度調整：</b> 原年終雙糧制度已調整為<b>「酌情花紅制度」</b>，與公司業績及個人表現掛鈎。<br>"
            "• <b>9月15日後入職：</b> 年終花紅由董事總經理按表現酌情處理。<br>"
            "• <b>詳細文件：</b> <span class='z-drive-path'>Z:\\Hrd-Public Folder\\16.0 人力資源政策及指引</span>"
        )
    },
    {
        "id": "facial_recognition_2026",
        "keywords": ["打卡", "拍卡", "人面識別", "外勤", "地盤打卡", "考勤"],
        "title": "📌 【最新考勤指引】人面識別 (2026年1月1日起生效)",
        "summary": (
            "• <b>簽到方式：</b> 上下班及外勤均須以<b>人面識別</b>完成「到場簽到、離場簽退」。<br>"
            "• <b>地盤要求：</b> 須於閘機打卡 + 手機人面識別「到達及離開各一次」。<br>"
            "• <b>修正截止：</b> 每月<b>「提交資料日」中午12:00前</b>須完成批核，逾期按系統紀錄扣假/扣薪。<br>"
            "• <b>詳細指引：</b> <span class='z-drive-path'>Z:\\Hrd-Public Folder\\16.0 人力資源政策及指引\\辦公時間及考勤指引 - 人面識別.pdf</span>"
        )
    },
    {
        "id": "training_subsidy_2025",
        "keywords": ["培訓", "進修", "資助", "學費", "上堂", "CPD", "退還資助", "服務期"],
        "title": "📌 【最新培訓資助政策】2025年11月1日起生效",
        "summary": (
            "• <b>申請類別：</b> 公司推薦（全額）／個人發展（通過試用期，按課程層級及預算決定）。<br>"
            "• <b>服務期承諾：</b> $1,000或以下→1個月，$15,001-$20,000→24個月。<br>"
            "• <b>提前離職：</b> 按未履行服務期比例退還資助金額。<br>"
            "• <b>申請表格：</b> HRF-011（個人）/ HRF-066（推薦）。完整文件：<span class='z-drive-path'>Z:\\Hrd-Public Folder\\16.0 人力資源政策及指引\\培訓資助政策.pdf</span>"
        )
    }
]

@st.cache_data(ttl=3600)
def load_policy_overrides():
    json_path = "policy_overrides.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if loaded:
                    return loaded
        except Exception as e:
            st.sidebar.warning(f"⚠️ 政策配置解析異常，使用內建值: {str(e)}")
    return DEFAULT_POLICY_OVERRIDES

def check_policy_overrides(query):
    overrides = load_policy_overrides()
    q_lower = query.lower()
    matched = []
    for ov in overrides:
        if any(kw in q_lower for kw in ov.get("keywords", [])):
            matched.append(ov)
    return matched

# ==========================================
# 3. 🌳 章節索引與智慧路由
# ==========================================
MANUAL_INDEX_TREE = {
    "chapters": [
        {"id": "ch3", "title": "第三章 僱傭條款", "intents": ["試用期", "調職", "離職", "終止合約", "解僱", "退休", "證明書"]},
        {"id": "ch4", "title": "第四章 辦公時間及考勤", "intents": ["上班時間", "考勤", "遲到", "早退", "拍卡", "人面識別", "打卡", "外勤", "地盤閘機", "惡劣天氣", "打風", "黑雨", "颱風"]},
        {"id": "ch5", "title": "第五章 薪酬管理", "intents": ["工資", "底薪", "出糧", "雙糧", "花紅", "加薪", "調薪", "酌情花紅"]},
        {"id": "ch6", "title": "第六章 假期", "intents": ["休息日", "公眾假期", "年假", "補假", "病假", "事假", "產假", "分娩假", "侍產假", "婚假", "恩恤假", "生日假", "無薪假"]},
        {"id": "ch7", "title": "第七章 員工福利及保障", "intents": ["強積金", "MPF", "醫療", "門診", "住院", "睇醫生", "康樂"]},
        {"id": "ch8", "title": "第八章 績效考核", "intents": ["考核", "評估", "appraisal", "最佳員工", "獎懲"]},
        {"id": "ch9", "title": "第九章 培訓和發展", "intents": ["培訓", "進修", "資助", "課程", "公司推薦", "個人發展", "服務承諾期", "退還資助", "學費"]},
        {"id": "ch10", "title": "第十章 紀律守則及防貪誠信守則", "intents": ["紀律", "行為", "防貪", "利益衝突", "賄賂", "收禮", "處分", "警告", "偽造紀錄"]},
        {"id": "appx", "title": "附則", "intents": ["保密協議", "電子通訊", "電腦使用"]}
    ],
    "keywords_to_chapter": {
        "辭職": "ch3", "通知期": "ch3", "試用": "ch3", 
        "打風": "ch4", "黑雨": "ch4", "ot": "ch4", "拍卡": "ch4", "人面識別": "ch4", "地盤打卡": "ch4",
        "雙糧": "ch5", "花紅": "ch5", "報稅": "ch5",
        "大假": "ch6", "al": "ch6", "sl": "ch6", "醫生紙": "ch6", "補假": "ch6", "請假": "ch6", "年假": "ch6",
        "claim錢": "ch7", "洗牙": "ch7", "津貼": "ch7",
        "升職": "ch8", 
        "上堂": "ch9", "培訓": "ch9", "資助": "ch9", "培訓資助": "ch9", "學費": "ch9", "進修": "ch9",
        "請客": "ch10", "利是": "ch10", "賭錢": "ch10", "保密": "appx"
    }
}

def analyze_intent_and_route(query, last_chapters=None):
    q_lower = query.lower()
    scores = {}
    
    # === 強制鎖定規則（解決關鍵字衝突） ===
    # 培訓相關 → 強制鎖定第九章
    if any(kw in q_lower for kw in ["培訓", "資助", "學費", "進修", "上堂", "CPD"]):
        return ["第九章 培訓和發展"]
    
    # 惡劣天氣 → 強制鎖定第四章
    if any(kw in q_lower for kw in ["打風", "黑雨", "颱風", "八號", "暴雨", "惡劣天氣"]):
        return ["第四章 辦公時間及考勤"]
    
    # === 加權計分 ===
    for kw, ch_id in MANUAL_INDEX_TREE["keywords_to_chapter"].items():
        if kw in q_lower:
            scores[ch_id] = scores.get(ch_id, 0) + 3
            
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
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# ==========================================
# 4. 🧠 精確頁碼映射與章節識別
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

def get_chapter_by_page_and_text(doc_page_num, text_content):
    actual_page = doc_page_num + 1
    
    # === 基於條文編號的精確判斷（優先級最高） ===
    # 第九章：培訓和發展
    if re.search(r'(9\.\d|第9章|培訓和發展)', text_content):
        return "第九章 培訓和發展"
    # 第三章：僱傭條款
    if re.search(r'(3\.\d|第3章|僱傭條款)', text_content):
        return "第三章 僱傭條款"
    # 第四章：辦公時間及考勤
    if re.search(r'(4\.\d|第4章|辦公時間|考勤)', text_content):
        return "第四章 辦公時間及考勤"
    # 第六章：假期
    if re.search(r'(6\.\d|第6章|假期|年假|病假)', text_content):
        return "第六章 假期"
    # 第十章：紀律守則
    if re.search(r'(10\.\d|第10章|紀律守則|防貪)', text_content):
        return "第十章 紀律守則及防貪誠信守則"
    
    # === 頁碼映射（備用） ===
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
    filename = uploaded_file.name
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n第", "\n\n", "\n", "。", "；", "，", " "]
        )
        
        raw_chunks = text_splitter.split_documents(documents)
        
        for chunk in raw_chunks:
            text = chunk.page_content.strip()
            
            # 雜訊過濾
            if len(text) < 50:
                continue
                
            # 移除頁碼/頁尾雜訊
            cleaned_text = re.sub(r'東淦工程有限公司\s*員工手冊\s*\d+\s*/\s*\d+', '', text)
            cleaned_text = re.sub(r'備忘\s*$', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'^第\s*\d+\s*頁\s*$', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = cleaned_text.strip()
            
            if len(cleaned_text) < 30:
                continue
                
            page_num = chunk.metadata.get("page", 0)
            chapter_title = get_chapter_by_page_and_text(page_num, cleaned_text)
            
            chunk.metadata["chapter"] = chapter_title
            chunk.metadata["source_file"] = filename
            chunk.metadata["page"] = page_num + 1
            chunk.page_content = f"[{chapter_title}]\n{cleaned_text}"
            chunks.append(chunk)
            
    except Exception as e:
        st.error(f"解析文件 {filename} 時出錯: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
    return chunks

# ==========================================
# 5. 主畫面佈局與狀態管理
# ==========================================
st.title("🏗️ 東淦工程有限公司 (Jumbo Orient)")
st.subheader("東淦員工手冊智能查詢系統 (jo-staff)")

st.info(
    "🔒 **內部數據安全保障：**\n"
    "- 本系統採用純本地數據比對技術，所有文件僅暫存於當前會話。\n"
    "- 點擊「清除對話」或關閉網頁時，所有紀錄將被徹底銷毀。"
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
        "請上傳《月薪員工手冊》PDF 檔案 (可多個)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清除對話", use_container_width=True):
            st.session_state.jo_messages = []
            st.session_state.last_chapters = []
            st.rerun()
    with col2:
        if st.button("🔄 重設全部", use_container_width=True):
            st.session_state.jo_messages = []
            st.session_state.last_chapters = []
            st.rerun()

if uploaded_files:
    for f in uploaded_files:
        all_chunks.extend(process_pdf_to_chunks(f))
        uploaded_file_names.append(f.name)
    
    if all_chunks:
        with st.spinner('構建雙引擎混合檢索矩陣中 (FAISS + BM25)...'):
            embeddings = get_embedding_model()
            vector_db = FAISS.from_documents(all_chunks, embeddings)
            faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
            
            bm25_retriever = BM25Retriever.from_documents(all_chunks)
            bm25_retriever.k = 5
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], 
                weights=[0.3, 0.7]
            )

with st.sidebar:
    st.markdown("---")
    st.header("📊 知識庫加載狀態")
    if uploaded_files:
        st.write(f"📁 已加載檔案數：{len(uploaded_files)} 份")
        st.write(f"🧩 有效解析段落：{len(all_chunks)} 段")
        st.success("✅ 雙引擎動態比對已啟動")
    else:
        st.write("尚未上傳檔案。")
        
    st.markdown("---")
    st.caption("📁 **完整政策目錄：** `Z:\\Hrd-Public Folder\\16.0 人力資源政策及指引`")
    st.caption("🌐 **公司網站：** [jumboorient.com.hk](https://jumboorient.com.hk/)")
    st.caption("⚙️ 如遇系統問題或特殊情境，請聯絡 [Jacky Law](https://jackylawck.github.io/jackylawck/) 。")

# ==========================================
# 6. 快捷提問區
# ==========================================
prompt = st.chat_input("請輸入您關於人事政策的疑問...")

if not prompt and (ensemble_retriever is not None):
    st.markdown("<div class='quick-btn-container'><b>💡 常見問題快速查詢：</b></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🌪️ 颱風黑雨安排", use_container_width=True): prompt = "八號風球或者黑雨，洗唔洗返工？"
    if col2.button("📷 人面識別考勤", use_container_width=True): prompt = "人面識別打卡點樣運作？外勤地盤點樣打卡？"
    if col3.button("🎓 培訓資助政策", use_container_width=True): prompt = "申請培訓資助有咩要求？離職要唔要退還資助？"
    if col4.button("🌴 有薪年假申請", use_container_width=True): prompt = "我有幾多日有薪年假？請假要提早幾多日申請？"

# ==========================================
# 7. 智能對話
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
            # 1. 最新政策檢查
            matched_overrides = check_policy_overrides(prompt)
            override_html = ""
            if matched_overrides:
                for ov in matched_overrides:
                    override_html += (
                        f"<div class='override-box'>"
                        f"<b>{ov['title']}</b><br><br>{ov['summary']}"
                        f"</div>"
                    )

            # 2. 章節路由
            routed_chapters = analyze_intent_and_route(prompt, st.session_state.last_chapters)
            st.session_state.last_chapters = routed_chapters
            
            enhanced_prompt = prompt
            routing_notice = ""
            if routed_chapters:
                chapters_str = "、".join(routed_chapters)
                enhanced_prompt = f"{prompt} 相關章節聚焦：{chapters_str}"
                routing_notice = f"<div class='routing-notice'>🔍 系統聚焦於：<b>{chapters_str}</b></div>"

            # 3. 混合檢索
            retrieved_docs = ensemble_retriever.invoke(enhanced_prompt)
            
            # 4. 對檢索結果進行章節驗證（防噪音）
            filtered_docs = []
            target_chapter = None
            if routed_chapters:
                target_chapter = routed_chapters[0]
                for doc in retrieved_docs:
                    doc_chapter = doc.metadata.get("chapter", "")
                    if target_chapter in doc_chapter or doc_chapter == target_chapter:
                        filtered_docs.append(doc)
            
            # 若過濾後無結果，退回全部檢索結果（但標註警告）
            if not filtered_docs:
                filtered_docs = retrieved_docs
                chapter_warning = True
            else:
                chapter_warning = False
            
            if filtered_docs:
                combined_content = ""
                seen_signatures = set()
                
                for doc in filtered_docs[:4]:
                    # 改良去重指紋
                    sig = re.sub(r'[，。、！？；：""''（）\s]+', '', doc.page_content)[:60]
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)
                    
                    chapter_tag = doc.metadata.get("chapter", "通用條文")
                    source_file = doc.metadata.get("source_file", "員工手冊")
                    page_num = doc.metadata.get("page", "?")
                    clean_content = doc.page_content.replace(f"[{chapter_tag}]\n", "").replace("\n", "<br>")
                    
                    combined_content += (
                        f"<span class='source-tag'>📍 來源：{source_file} - {chapter_tag} (第{page_num}頁)</span>"
                        f"{clean_content}<br><hr><br>"
                    )

                # 若未過濾成功，加註提示
                if chapter_warning and routed_chapters:
                    combined_content = (
                        f"<i>⚠️ 系統未能精確定位至「{target_chapter}」的條文，以下為相關參考：</i><br><br>"
                        + combined_content
                    )

                st.markdown(f"<div class='confidence-badge'>✅ 綜合查詢結果</div>", unsafe_allow_html=True)
                
                response_html = (
                    f"{routing_notice}"
                    f"{override_html}"
                    f"<div class='answer-box'>"
                    f"<b>📋 手冊條文參考：</b><br><br>{combined_content}"
                    f"</div>"
                    f"<small><i>※ 提示：最新政策請以頂部通告為準。完整文件請至 <span class='z-drive-path'>Z:\\Hrd-Public Folder\\16.0 人力資源政策及指引</span> 查閱。</i></small>"
                )
                
                st.markdown(response_html.replace(routing_notice, ""), unsafe_allow_html=True) 
                st.session_state.jo_messages.append({"role": "assistant", "content": response_html})
            else:
                fallback_msg = "抱歉，在手冊中找不到相關條文。建議換個說法，或聯絡人力資源組。"
                st.warning(fallback_msg)
                st.session_state.jo_messages.append({"role": "assistant", "content": fallback_msg})
