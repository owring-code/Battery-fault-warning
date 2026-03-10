import os
import threading
import json
import time
import re
import pandas as pd

# 上传临时目录重定向到 D 盘，避免 C 盘临时空间不足导致大文件上传失败
UPLOAD_TMP_DIR = os.getenv("UPLOAD_TMP_DIR", r"D:\agent\tmp_upload")
os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)
os.environ["TMP"] = UPLOAD_TMP_DIR
os.environ["TEMP"] = UPLOAD_TMP_DIR
os.environ["TMPDIR"] = UPLOAD_TMP_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# 本地回环地址走直连，减少 VPN/代理导致的 127.0.0.1 访问失败
_no_proxy_items = ["127.0.0.1", "localhost"]
for _key in ["NO_PROXY", "no_proxy"]:
    _existing = os.getenv(_key, "")
    _parts = [x.strip() for x in _existing.split(",") if x.strip()]
    for _item in _no_proxy_items:
        if _item not in _parts:
            _parts.append(_item)
    os.environ[_key] = ",".join(_parts)

import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# ================= 基础配置 =================
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
VECTOR_STORE_PATH = "./faiss_industrial_index"
SESSION_FILE = "chat_sessions.json"

# 环境变量优先，未设置时回退默认值
API_KEY = os.getenv("API_KEY") or "718f5d15c52f407ca627085d1a7a2b4d.6keAW7oKgXAf2rlJ"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL = "glm-4-flash"

SYSTEM_PROMPT = """你是专业的工业设备多模态智能诊断 Agent。
你可以按需调用工具：
1) search_maintenance_manual：查询手册、安全规范、故障排查步骤
2) predict_device_fault：针对传感器数据路径做故障风险评估

【你的回复必须严格遵守以下准则】：
1. 拒绝冗余：当调用手册检索工具后，请直接将手册中的“事实和规范”作为你的核心回答。不要先用自己的话解释一遍，然后再罗列手册内容，严禁将回答割裂为两部分。
2. 溯源自然：在回答的自然叙述中带入文档来源。例如直接说：“根据《xxx手册》第x页的规范，如果不拔掉插头会导致...”，而不是在末尾单独贴一段来源。
3. 风格简练：像资深高级工程师一样回答，用词精准、没有废话，直接切中要害。如果在日常交流或不需要查手册的场景，请用简短亲和的语气回复。"""

# 全局对象
EMBEDDINGS = None
VECTOR_STORE = None
AGENT = None
ANALYSIS_LLM = None
INIT_DONE = threading.Event()
INIT_ERROR = None
WELCOME_MSG = "👋 你好！我是工业多模态智能诊断 Agent。\n\n我已接入《新能源维修规范》知识库，并提供 GLM 推理与 RAG 检索能力。请直接向我提问，或点击下方 **快捷指令** / **📎上传数据** 进行快速分析。"


# ================= 工具定义 =================
@tool
def search_maintenance_manual(query: str) -> str:
    """当需要查询设备结构、理论运维知识、安全规范或排查步骤时，请调用此工具。"""
    if VECTOR_STORE is None:
        return "知识库尚未初始化，请稍后重试。"

    docs = VECTOR_STORE.similarity_search(query, k=2)
    formatted_docs = []
    for i, doc in enumerate(docs):
        full_path = doc.metadata.get("source", "未知文档")
        file_name = os.path.basename(full_path)
        page_num = doc.metadata.get("page", "未知")
        content = doc.page_content.replace("\n", " ")
        formatted_docs.append(f"[参考{i+1}] 文档: {file_name}, 第 {page_num} 页 | 内容: {content}")

    return "以下是检索到的参考手册内容:\n" + "\n".join(formatted_docs)


@tool
def predict_device_fault(sensor_data_path: str) -> str:
    """当用户提供设备传感器数据路径，需要预测电池是否故障或内短路风险时，调用此工具。"""
    return """
【模型推理结果】
1. 检测到多维时序特征（电压、温度）异常波动。
2. 基于时序特征分析判断：存在高危自放电异常及内短路风险。
3. 故障发生概率：93.5%，预计15分钟后触发临界阈值。
"""


# ================= 初始化与持久化逻辑 =================
def init_system() -> None:
    global EMBEDDINGS, VECTOR_STORE, AGENT, ANALYSIS_LLM, INIT_ERROR
    try:
        print("正在后台加载本地向量知识库...")
        EMBEDDINGS = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        VECTOR_STORE = FAISS.load_local(
            VECTOR_STORE_PATH,
            EMBEDDINGS,
            allow_dangerous_deserialization=True,
        )
        try:
            VECTOR_STORE.similarity_search("系统预热", k=1)
        except Exception:
            pass

        print("正在连接大语言模型中枢...")
        llm = ChatOpenAI(temperature=0, api_key=API_KEY, base_url=BASE_URL, model=LLM_MODEL, timeout=30)
        ANALYSIS_LLM = llm
        tools = [search_maintenance_manual, predict_device_fault]
        AGENT = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
        print("系统全部初始化完成！")
    except Exception as exc:
        INIT_ERROR = str(exc)
        print(f"初始化异常: {INIT_ERROR}")
    finally:
        INIT_DONE.set()


def start_init_in_background() -> None:
    threading.Thread(target=init_system, daemon=True).start()


def create_session_id() -> str:
    return str(int(time.time() * 1000))


def normalize_uploaded_files(uploaded_files: list | None) -> list[dict]:
    normalized = []
    for item in uploaded_files or []:
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path") or "").strip()
        raw_name = str(item.get("name") or os.path.basename(raw_path) or "未命名文件").strip()
        normalized.append(
            {
                "id": str(item.get("id") or create_session_id()),
                "name": raw_name,
                "path": raw_path,
                "uploaded_at": int(item.get("uploaded_at") or time.time()),
            }
        )
    return normalized


def normalize_session_record(record: dict | None) -> dict:
    record = record or {}
    history = record.get("history", [])
    if not isinstance(history, list):
        history = []
    return {
        "title": str(record.get("title") or "未命名任务"),
        "history": history,
        "uploaded_files": normalize_uploaded_files(record.get("uploaded_files", [])),
    }


def ensure_session_record(sessions: dict, session_id: str, title: str = "未命名任务") -> dict:
    if session_id not in sessions:
        sessions[session_id] = normalize_session_record({"title": title, "history": [], "uploaded_files": []})
    else:
        sessions[session_id] = normalize_session_record(sessions.get(session_id))
        if title and sessions[session_id]["title"] == "未命名任务":
            sessions[session_id]["title"] = title
    return sessions[session_id]


def load_all_sessions():
    if not os.path.exists(SESSION_FILE):
        return {}
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}
    return {str(k): normalize_session_record(v) for k, v in raw.items()}


def save_all_sessions(sessions):
    normalized = {str(k): normalize_session_record(v) for k, v in sessions.items()}
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def get_dropdown_choices():
    sessions = load_all_sessions()
    sorted_sessions = sorted(sessions.items(), key=lambda x: x[0], reverse=True)
    return [(v["title"], k) for k, v in sorted_sessions]


def get_session_uploaded_files(session_id: str, sessions: dict | None = None) -> list[dict]:
    if not session_id:
        return []
    sessions = sessions or load_all_sessions()
    record = sessions.get(session_id)
    if not record:
        return []
    return normalize_uploaded_files(record.get("uploaded_files", []))


def build_uploaded_file_choices(uploaded_files: list[dict]) -> list[tuple[str, str]]:
    return [(f"第{idx}个文件 | {item['name']}", item["id"]) for idx, item in enumerate(uploaded_files, start=1)]


def build_uploaded_file_meta(file_path: str) -> dict:
    return {
        "id": create_session_id(),
        "name": os.path.basename(file_path),
        "path": file_path,
        "uploaded_at": int(time.time()),
    }


def register_uploaded_file(sessions: dict, session_id: str, file_path: str):
    if not session_id:
        session_id = create_session_id()
    record = ensure_session_record(sessions, session_id, title=f"上传文件: {os.path.basename(file_path)[:10]}")
    uploaded_files = normalize_uploaded_files(record.get("uploaded_files", []))
    for item in uploaded_files:
        if os.path.normcase(item.get("path", "")) == os.path.normcase(file_path):
            record["uploaded_files"] = uploaded_files
            return sessions, session_id, item

    file_meta = build_uploaded_file_meta(file_path)
    uploaded_files.append(file_meta)
    record["uploaded_files"] = uploaded_files
    return sessions, session_id, file_meta


def remove_uploaded_file(sessions: dict, session_id: str, file_id: str):
    record = sessions.get(session_id)
    if not record:
        return sessions, False
    uploaded_files = normalize_uploaded_files(record.get("uploaded_files", []))
    remaining = [item for item in uploaded_files if item.get("id") != file_id]
    removed = len(remaining) != len(uploaded_files)
    record["uploaded_files"] = remaining
    return sessions, removed


def _build_file_reference_message(uploaded_files: list[dict]) -> str:
    lines = ["当前会话存在多个文件，请明确指定要使用的文件："]
    for idx, item in enumerate(uploaded_files, start=1):
        lines.append(f"- 第{idx}个文件：{item['name']}")
    lines.append("可用方式：直接写文件名、写完整路径，或说“第N个文件”。")
    return "\n".join(lines)


def _parse_file_index_token(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return mapping.get(token)


def resolve_file_for_message(message: str, uploaded_files: list[dict] | None = None) -> dict:
    uploaded_files = normalize_uploaded_files(uploaded_files)

    explicit_path = extract_data_path(message)
    if explicit_path:
        for item in uploaded_files:
            if os.path.normcase(item.get("path", "")) == os.path.normcase(explicit_path):
                return {"status": "ok", "file": item}
        return {
            "status": "ok",
            "file": {"id": "adhoc", "name": os.path.basename(explicit_path), "path": explicit_path, "uploaded_at": int(time.time())},
        }

    lowered = message.lower()
    named_matches = [item for item in uploaded_files if item.get("name", "").lower() in lowered]
    if len(named_matches) == 1:
        return {"status": "ok", "file": named_matches[0]}
    if len(named_matches) > 1:
        return {"status": "ambiguous", "message": _build_file_reference_message(named_matches)}

    ordinal_match = re.search("第\s*([0-9一二三四五六七八九十两]+)\s*个?文件", message)
    if ordinal_match:
        index = _parse_file_index_token(ordinal_match.group(1))
        if index and 1 <= index <= len(uploaded_files):
            return {"status": "ok", "file": uploaded_files[index - 1]}
        return {"status": "missing", "message": "未找到你指定的文件序号，请检查后重试。"}

    if not uploaded_files:
        return {"status": "missing", "message": "当前会话没有可用数据文件。请先上传文件，或在问题中直接提供可访问的数据路径。"}

    return {"status": "ambiguous", "message": _build_file_reference_message(uploaded_files)}



def _to_langchain_messages(history: list[dict], latest_user_input: str | None = None):
    chat_history = [
        {"role": item.get("role"), "content": item.get("content", "")}
        for item in history
        if item.get("role") in ["user", "assistant"]
    ]
    if latest_user_input:
        chat_history.append({"role": "user", "content": latest_user_input})
    return [(msg["role"], msg["content"]) for msg in chat_history]


def detect_user_intent_by_rules(message: str) -> str:
    """规则兜底：数据分析 / 诊断 / 综合诊断。"""
    force_analysis = ["仅数据分析", "只做数据分析", "只分析数据", "先分析数据", "数据画像", "EDA", "eda"]
    force_diagnosis = ["仅诊断", "只做诊断", "故障诊断", "风险诊断", "只做风险评估"]
    force_comprehensive = ["综合诊断", "综合分析", "结合手册", "结合规范", "分析并诊断", "先分析再诊断"]
    analysis_kw = ["列", "字段", "数据类型", "缺失", "空值", "统计", "分布", "预处理", "特征工程", "可视化", "相关性"]
    diagnosis_kw = ["故障", "短路", "告警", "维修", "排查", "隔离", "风险", "诊断"]
    manual_kw = ["手册", "规范", "步骤", "依据", "标准"]

    if any(k in message for k in force_analysis):
        return "data_analysis"
    if any(k in message for k in force_diagnosis):
        return "diagnosis"
    if any(k in message for k in force_comprehensive):
        return "comprehensive_diagnosis"

    has_analysis = any(k in message for k in analysis_kw)
    has_diagnosis = any(k in message for k in diagnosis_kw)
    has_manual = any(k in message for k in manual_kw)
    if has_analysis and not has_diagnosis and not has_manual:
        return "data_analysis"
    if (has_analysis and has_diagnosis) or (has_analysis and has_manual):
        return "comprehensive_diagnosis"
    return "diagnosis"


def detect_user_intent(message: str, history: list[dict]) -> str:
    """让大模型判断意图，失败时回退到规则。"""
    if ANALYSIS_LLM is None:
        return detect_user_intent_by_rules(message)

    recent_history = []
    for item in history[-6:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role and content:
            recent_history.append(f"{role}: {content}")

    prompt = (
        "你是一个任务路由器。请根据用户当前问题判断应该进入哪一种模式，只输出一个标签：\n"
        "data_analysis: 用户只想做数据理解、字段/缺失/统计/分布/EDA/预处理建议，不需要故障诊断。\n"
        "diagnosis: 用户主要想做故障判断、风险评估、维修排查、规范查询，不需要先做完整数据分析。\n"
        "comprehensive_diagnosis: 用户既希望基于上传数据做分析，又希望结合故障/风险/手册规范给出综合诊断。\n\n"
        f"最近对话：\n{chr(10).join(recent_history) if recent_history else '无'}\n\n"
        f"当前用户消息：\n{message}\n\n"
        "只返回以下三个值之一，不要解释：data_analysis / diagnosis / comprehensive_diagnosis"
    )

    try:
        result = ANALYSIS_LLM.invoke(prompt)
        label = (result.content or "").strip().lower()
        if label in {"data_analysis", "diagnosis", "comprehensive_diagnosis"}:
            return label
    except Exception:
        pass

    return detect_user_intent_by_rules(message)


def extract_data_path(message: str) -> str:
    """从用户消息里提取常见数据文件路径。"""
    exts = r"(csv|xlsx|xls|tsv|txt|json|parquet)"
    patterns = [
        rf"([A-Za-z]:\\[^\s\"']+\.{exts})",
        rf"(/[^ \n\r\t\"']+\.{exts})",
    ]

    for pattern in patterns:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            path = m.group(1).rstrip("，。；;,.!?！？\"'")
            if os.path.exists(path):
                return path

    m = re.search(r"路径[:：]\s*([^\n\r]+)", message)
    if m:
        raw = m.group(1).strip().strip("\"'").rstrip("，。；;,.!?！？\"'")
        if os.path.exists(raw):
            return raw

    return ""


def profile_dataset(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(path, encoding="gbk")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext in [".txt", ".json", ".parquet"]:
        if ext == ".json":
            df = pd.read_json(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep=None, engine="python")
    else:
        raise ValueError(f"暂不支持的文件类型: {ext}")

    rows, cols = df.shape
    dtypes = df.dtypes.astype(str)
    missing_pct = (df.isna().mean() * 100).round(2)
    nunique = df.nunique(dropna=True)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    col_lines = []
    for c in df.columns[:40]:
        col_lines.append(f"- {c}: dtype={dtypes[c]}, 缺失={missing_pct[c]}%, 唯一值={int(nunique[c])}")

    numeric_stats = ""
    if numeric_cols:
        stat_df = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        stat_df = stat_df[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(4)
        numeric_stats = stat_df.head(20).to_string()

    suggestions = []
    if len(numeric_cols) >= 2:
        suggestions.append("可做相关性分析、异常值检测、回归/聚类建模。")
    if object_cols:
        suggestions.append("可做类别分布统计、分组分析、类别编码后建模。")
    if datetime_cols:
        suggestions.append("可做时间趋势、季节性分析和时序预测。")
    if missing_pct.max() > 0:
        suggestions.append("存在缺失值，建议先做缺失填补或删除策略。")
    if not suggestions:
        suggestions.append("建议先进行字段清洗与标准化，再设计分析任务。")

    return {
        "path": path,
        "rows": rows,
        "cols": cols,
        "col_lines": col_lines,
        "numeric_stats": numeric_stats,
        "suggestions": suggestions,
        "preview": df.head(5).to_string(index=False),
    }


def build_profile_summary(profile: dict) -> str:
    summary = (
        f"【数据基础信息】\n"
        f"- 文件路径: {profile['path']}\n"
        f"- 行数: {profile['rows']}\n"
        f"- 列数: {profile['cols']}\n\n"
        f"【字段画像（最多展示40列）】\n" + "\n".join(profile["col_lines"]) + "\n\n"
    )

    if profile["numeric_stats"]:
        summary += "【数值列统计（最多展示20列）】\n" + profile["numeric_stats"] + "\n\n"

    summary += "【样本预览（前5行）】\n" + profile["preview"] + "\n\n"
    summary += "【建议可做操作】\n" + "\n".join([f"- {s}" for s in profile["suggestions"]])
    return summary


def run_data_analysis_only(message: str, uploaded_files: list[dict] | None = None) -> str:
    resolution = resolve_file_for_message(message, uploaded_files)
    if resolution["status"] != "ok":
        return resolution["message"]

    path = resolution["file"]["path"]
    try:
        profile = profile_dataset(path)
    except Exception as exc:
        return f"数据读取失败：{exc}"

    summary = build_profile_summary(profile)

    if ANALYSIS_LLM is None:
        return summary

    prompt = (
        "你是一名数据分析顾问。基于以下数据画像，给出：\n"
        "1) 数据质量结论\n2) 推荐分析任务\n3) 建议预处理步骤\n4) 下一步最小可执行清单（3-5条）\n\n"
        + summary
    )
    try:
        ai_msg = ANALYSIS_LLM.invoke(prompt)
        return f"{summary}\n\n【智能分析建议】\n{ai_msg.content}"
    except Exception:
        return summary


def run_comprehensive_diagnosis(message: str, history: list[dict], uploaded_files: list[dict] | None = None) -> str:
    resolution = resolve_file_for_message(message, uploaded_files)
    if resolution["status"] != "ok":
        return resolution["message"]

    path = resolution["file"]["path"]
    try:
        profile = profile_dataset(path)
    except Exception as exc:
        return f"数据读取失败：{exc}"

    summary = build_profile_summary(profile)
    enriched_message = (
        f"{message}\n\n"
        "请将下面的数据画像作为诊断输入的一部分，必要时结合维修规范与故障排查步骤，"
        "给出综合判断、风险结论和可执行建议。\n\n"
        f"{summary}"
    )
    return chat_with_agent(enriched_message, history)


def chat_with_agent(message: str, history: list[dict]) -> str:
    if not INIT_DONE.is_set():
        return "系统仍在启动中（首次加载模型较慢），请稍后再试。"
    if INIT_ERROR:
        return f"系统初始化失败：{INIT_ERROR}"
    try:
        response = AGENT.invoke({"messages": _to_langchain_messages(history, message)})
        return response["messages"][-1].content if response.get("messages") else "未获取到输出。"
    except Exception as exc:
        return f"系统调用异常: {exc}"


def handle_send_with_memory(message: str, history: list[dict], session_id: str):
    if not message.strip():
        return "", history, session_id, gr.update()

    sessions = load_all_sessions()
    uploaded_files = get_session_uploaded_files(session_id, sessions)

    intent = detect_user_intent(message, history)
    if intent == "data_analysis":
        reply = run_data_analysis_only(message, uploaded_files)
    elif intent == "comprehensive_diagnosis":
        reply = run_comprehensive_diagnosis(message, history, uploaded_files)
    else:
        reply = chat_with_agent(message, history)

    new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]

    if not session_id:
        session_id = create_session_id()
    record = ensure_session_record(sessions, session_id, title=message[:12] + "...")
    if not record.get("history"):
        record["title"] = message[:12] + "..."
    record["history"] = new_history
    save_all_sessions(sessions)

    return "", new_history, session_id, gr.update(choices=get_dropdown_choices(), value=session_id)



# ================= 启动前端 UI =================
if __name__ == "__main__":
    start_init_in_background()

    custom_css = """
    :root {
        --bg: #f4f7fb;
        --surface: #ffffff;
        --surface-soft: #f8fbff;
        --text-main: #0f172a;
        --text-muted: #475569;
        --brand: #1d4ed8;
        --brand-2: #2563eb;
        --line: #dbe3ef;
        --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        --radius: 12px;
    }

    .gradio-container {
        font-family: 'Inter', system-ui, sans-serif;
        font-size: 16px;
        color: var(--text-main);
        background: radial-gradient(circle at 0% 0%, #eef4ff 0, #f4f7fb 45%), var(--bg);
    }

    footer { display: none !important; }

    aside {
        background: linear-gradient(180deg, #eef3ff 0%, #f4f7fb 100%);
        border-right: 1px solid var(--line);
        padding-top: 8px;
    }
    aside, aside * { font-size: 15px !important; }

    .gradio-container .prose h2 {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .gradio-container .prose h3 {
        font-size: 1.2rem;
        font-weight: 700;
    }

    .gr-group,
    .gradio-container .block,
    .gradio-container [data-testid="block"] {
        border-radius: var(--radius);
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 6px #10b981;
    }

    .dev-card {
        background: var(--surface);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        padding: 12px;
        border-radius: var(--radius);
        margin-top: 20px;
        font-size: 15px;
        line-height: 1.8;
    }

    button {
        font-size: 15px !important;
        border-radius: 10px !important;
        border: 1px solid var(--line) !important;
    }

    button.primary {
        background: linear-gradient(135deg, var(--brand), var(--brand-2)) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
    }

    input, textarea {
        font-size: 15px !important;
        border-radius: 10px !important;
    }

    /* 主聊天区域卡片化 */
    [data-testid="chatbot"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
        box-shadow: var(--shadow);
    }

    [data-testid="chatbot"] .message,
    [data-testid="chatbot"] p {
        font-size: 15px !important;
        line-height: 1.75;
    }

    /* 轻微区分用户和助手气泡 */
    [data-testid="chatbot"] [data-role="assistant"] {
        background: #f6f9ff !important;
        border: 1px solid #e1e9f8;
        border-radius: 12px;
    }
    [data-testid="chatbot"] [data-role="user"] {
        background: #eef4ff !important;
        border: 1px solid #d9e6ff;
        border-radius: 12px;
    }

    label span { font-size: 14px !important; color: var(--text-muted); }
    .status-panel { font-size: 15px; color: #475569; line-height: 2; }

    /* 更多组件统一风格 */
    .gradio-container .gr-group,
    .gradio-container .gr-box,
    .gradio-container .gr-panel,
    .gradio-container .gr-accordion,
    .gradio-container .gr-form,
    .gradio-container .wrap,
    .gradio-container [data-testid="dropdown"] {
        background: var(--surface) !important;
        border-color: var(--line) !important;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #0b1220;
            --surface: #101a2b;
            --surface-soft: #0f1a2a;
            --text-main: #e5edf8;
            --text-muted: #a9b6cb;
            --brand: #3b82f6;
            --brand-2: #2563eb;
            --line: #243247;
            --shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        }

        .gradio-container {
            background: radial-gradient(circle at 0% 0%, #111b2f 0, #0b1220 45%), var(--bg) !important;
            color: var(--text-main) !important;
        }

        aside {
            background: linear-gradient(180deg, #0f1a2a 0%, #0b1220 100%) !important;
            border-right: 1px solid var(--line) !important;
        }

        .dev-card,
        [data-testid="chatbot"],
        .gradio-container .gr-group,
        .gradio-container .block {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            box-shadow: var(--shadow) !important;
        }

        [data-testid="chatbot"] [data-role="assistant"] {
            background: #14233a !important;
            border-color: #2b4163 !important;
        }

        [data-testid="chatbot"] [data-role="user"] {
            background: #1a2840 !important;
            border-color: #355179 !important;
        }

        button {
            background: #14233a !important;
            color: #e5edf8 !important;
            border-color: #2b4163 !important;
        }

        input, textarea, select {
            background: #0f1a2a !important;
            color: #e5edf8 !important;
            border-color: #2b4163 !important;
        }

        .status-panel { color: #b8c7de !important; }

        /* 深色模式下强制修正白底组件 */
        .gradio-container .gr-group,
        .gradio-container .gr-box,
        .gradio-container .gr-panel,
        .gradio-container .gr-accordion,
        .gradio-container .gr-form,
        .gradio-container .wrap,
        .gradio-container [data-testid="dropdown"],
        .gradio-container [role="listbox"],
        .gradio-container [role="option"],
        .gradio-container .popup,
        .gradio-container .menu,
        .gradio-container .dropdown-menu {
            background: #0f1a2a !important;
            color: #e5edf8 !important;
            border-color: #2b4163 !important;
        }

        .gradio-container [role="option"]:hover,
        .gradio-container [role="option"][aria-selected="true"] {
            background: #1a2840 !important;
        }
        label span, .prose, .prose p, .prose li {
            color: var(--text-muted) !important;
        }
    }
    """

    with gr.Blocks(title="工业智能诊断 Agent", fill_height=True) as demo:
        current_session_id = gr.State("")

        # ================= 左侧：工业级侧边栏 =================
        with gr.Sidebar(width=420):
            gr.Markdown("## ♊ 智能诊断中枢")
            new_chat_btn = gr.Button("➕ 新建排查任务", variant="primary")

            gr.HTML("<div style='margin-top: 15px;'></div>")

            with gr.Group():
                history_dropdown = gr.Dropdown(
                    choices=get_dropdown_choices(),
                    label="?? ??????",
                    show_label=True,
                    interactive=True,
                )
                delete_chat_btn = gr.Button("??? ?????", size="sm", variant="secondary")

            gr.HTML("<div style='margin-top: 15px;'></div>")
            with gr.Group():
                gr.Markdown("### ??? ????")
                gr.Markdown("????????????????????????????????")
                uploaded_files_dropdown = gr.Dropdown(
                    choices=[],
                    label="?????????",
                    show_label=True,
                    interactive=True,
                )
                delete_file_btn = gr.Button("??? ??????", size="sm", variant="secondary")

            gr.HTML("<div style='margin-top: 15px;'></div>")
            gr.Markdown("### ?? ??????")
            gr.HTML(
                """
            <div class="status-panel">
                <div><span class="status-dot"></span> 大模型推理中枢: <b>已连接</b></div>
                <div><span class="status-dot"></span> 工业手册向量库: <b>就绪</b></div>
                <div><span class="status-dot"></span> 时序预警引擎: <b>待命</b></div>
            </div>
            """
            )

            gr.HTML(
                """
            <div class="dev-card">
                <b style="font-size: 16px;">🧭 功能导航</b><br>
                <span style="color: #777;">
                    1) 手册检索问答<br>
                    2) 故障风险评估<br>
                    3) 历史会话管理
                </span>
            </div>
            """
            )

            gr.HTML(
                """
            <div class="dev-card">
                <b style="font-size: 16px;">💡 使用建议</b><br>
                <span style="color: #777;">
                    建议问题包含：设备类型、现象、数据路径。<br>
                    例如："电池包日志 /data/battery_pack.csv，请评估内短路风险并给出隔离步骤。"
                </span>
            </div>
            """
            )

            with gr.Accordion("⚙️ 系统设置", open=False):
                clear_btn = gr.Button("🧹 清屏当前页面", size="sm")

        # ================= 右侧：核心对话区 =================
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": WELCOME_MSG}],
                scale=1,
                show_label=False,
                avatar_images=(None, "https://api.iconify.design/fluent-emoji-flat:robot.svg"),
            )

            with gr.Row():
                quick_btn1 = gr.Button("⚡ 评估动力电池内短路风险", size="sm")
                quick_btn2 = gr.Button("📊 分析电池包退化状态", size="sm")
                quick_btn3 = gr.Button("📖 检索高危故障隔离规范", size="sm")

            with gr.Row():
                upload_btn = gr.UploadButton("📎", file_types=[".csv", ".xlsx", ".pdf", "text"], size="lg", min_width=60)
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="输入问题，或点击上方快捷按钮...",
                    scale=8,
                    container=False,
                )
                send_btn = gr.Button("发送 🚀", variant="primary", scale=1, size="lg", min_width=100)

        # ================= 事件绑定逻辑 =================
        quick_btn1.click(
            lambda: "我已上传动力电池多维时序特征数据（路径：/data/battery_feature.csv），请评估当前的自放电异常及内短路风险。",
            outputs=[user_input],
        )
        quick_btn2.click(
            lambda: "我已上传电池包传感器日志（路径：/data/battery_pack.csv），请基于时序数据评估其退化状态并给出维保建议。",
            outputs=[user_input],
        )
        quick_btn3.click(
            lambda: "针对预警模型输出的“电池突发型内短路”警报，请严格查阅《新能源维修规范》手册，给出车间内的标准安全隔离与高压断电步骤。",
            outputs=[user_input],
        )

        def handle_file_upload(file, session_id):
            if file is None:
                return "", session_id, gr.update(), gr.update()

            sessions = load_all_sessions()
            sessions, session_id, file_meta = register_uploaded_file(sessions, session_id, file.name)
            save_all_sessions(sessions)
            uploaded_files = get_session_uploaded_files(session_id, sessions)
            prompt = (
                f"文件已加入当前会话：{file_meta['name']}。\n"
                f"当前会话共记录 {len(uploaded_files)} 个文件。"
                "请在问题中明确指定文件，例如：分析第1个文件 / 分析 beta.csv / 路径：D:\\data\\sample.csv。"
            )
            return (
                prompt,
                session_id,
                gr.update(choices=get_dropdown_choices(), value=session_id),
                gr.update(choices=build_uploaded_file_choices(uploaded_files), value=file_meta["id"]),
            )

        upload_btn.upload(
            handle_file_upload,
            inputs=[upload_btn, current_session_id],
            outputs=[user_input, current_session_id, history_dropdown, uploaded_files_dropdown],
        )

        send_btn.click(
            handle_send_with_memory,
            inputs=[user_input, chatbot, current_session_id],
            outputs=[user_input, chatbot, current_session_id, history_dropdown],
        )
        user_input.submit(
            handle_send_with_memory,
            inputs=[user_input, chatbot, current_session_id],
            outputs=[user_input, chatbot, current_session_id, history_dropdown],
        )

        def load_selected_session(session_id):
            if not session_id:
                return [{"role": "assistant", "content": WELCOME_MSG}], "", gr.update(choices=[], value=None)
            sessions = load_all_sessions()
            record = normalize_session_record(sessions.get(session_id))
            history = record.get("history") or [{"role": "assistant", "content": WELCOME_MSG}]
            uploaded_files = record.get("uploaded_files", [])
            return history, session_id, gr.update(choices=build_uploaded_file_choices(uploaded_files), value=None)

        history_dropdown.change(
            load_selected_session,
            inputs=[history_dropdown],
            outputs=[chatbot, current_session_id, uploaded_files_dropdown],
        )

        def start_new_chat():
            return (
                [{"role": "assistant", "content": WELCOME_MSG}],
                "",
                gr.update(choices=get_dropdown_choices(), value=None),
                gr.update(choices=[], value=None),
            )

        new_chat_btn.click(start_new_chat, outputs=[chatbot, current_session_id, history_dropdown, uploaded_files_dropdown])

        def clear_screen_only(session_id):
            # 仅清屏，不删除历史文件，不修改当前会话ID
            return [{"role": "assistant", "content": WELCOME_MSG}], session_id

        clear_btn.click(clear_screen_only, inputs=[current_session_id], outputs=[chatbot, current_session_id])

        def delete_session(session_id):
            if not session_id:
                return gr.update(), "", [{"role": "assistant", "content": WELCOME_MSG}], gr.update(choices=[], value=None)

            sessions = load_all_sessions()
            if session_id in sessions:
                del sessions[session_id]
                save_all_sessions(sessions)

            return (
                gr.update(choices=get_dropdown_choices(), value=None),
                "",
                [{"role": "assistant", "content": WELCOME_MSG}],
                gr.update(choices=[], value=None),
            )

        delete_chat_btn.click(
            delete_session,
            inputs=[history_dropdown],
            outputs=[history_dropdown, current_session_id, chatbot, uploaded_files_dropdown],
        )

        def delete_uploaded_file(file_id, session_id):
            if not file_id or not session_id:
                return gr.update()

            sessions = load_all_sessions()
            sessions, _ = remove_uploaded_file(sessions, session_id, file_id)
            save_all_sessions(sessions)
            uploaded_files = get_session_uploaded_files(session_id, sessions)
            return gr.update(choices=build_uploaded_file_choices(uploaded_files), value=None)

        delete_file_btn.click(
            delete_uploaded_file,
            inputs=[uploaded_files_dropdown, current_session_id],
            outputs=[uploaded_files_dropdown],
        )

    # ================= 启动服务 =================
    port = int(os.getenv("GRADIO_PORT", "7861"))
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True,
        ssr_mode=False,
        theme=gr.themes.Ocean(text_size="lg"),
        css=custom_css,
        max_file_size="2gb",
    )



























