import os
import sys
import io
import json
import sqlite3
import numpy as np
import onnxruntime as ort
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# ================= 1. 全局配置 =================
load_dotenv()
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("❌ 未检测到 DEEPSEEK_API_KEY 环境变量")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

BASE_DIR = r'C:\Users\binx\Desktop\ultralytics-yolo11-main11\Bearing'
ONNX_PATH = os.path.join(BASE_DIR, 'bearing_autoencoder.onnx')
MANUAL_PATH = os.path.join(BASE_DIR, 'bearing_manual.txt')
DB_PATH = os.path.join(BASE_DIR, 'work_order_history.db')
THRESHOLD = 0.01

# ================= 2. 初始化 ONNX 推理引擎 =================
print("✅ 正在加载 ONNX 推理引擎...")
ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

def detect_anomaly(signal_window):
    input_data = signal_window.reshape(1, 1, 1024).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    reconstructed = ort_session.run(None, ort_inputs)[0]
    return float(np.mean((reconstructed - input_data) ** 2))

# ================= 3. 初始化 RAG 知识库 =================
print("✅ 正在加载本地 RAG 知识库...")
default_ef = embedding_functions.ONNXMiniLM_L6_V2()
chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection(name="bearing_manual")
except Exception:
    pass
collection = chroma_client.create_collection(name="bearing_manual", embedding_function=default_ef)

with open(MANUAL_PATH, 'r', encoding='utf-8') as f:
    manual_text = f.read()

chunks = [chunk.strip() for chunk in manual_text.split('[手册条款') if chunk.strip()]
chunks = ['[手册条款' + c for c in chunks]
for i, chunk in enumerate(chunks):
    collection.add(documents=[chunk], metadatas=[{"source": f"手册条款_{i+1}"}], ids=[f"id_{i+1}"])
print(f"✅ RAG 知识库就绪，共 {len(chunks)} 条条款")

# ================= 4. 初始化历史工单数据库 =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS work_orders
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  machine_id TEXT,
                  error_value REAL,
                  report TEXT)''')
    conn.commit()
    conn.close()

init_db()
print("✅ 历史工单数据库就绪")

# ================= 5. 工具函数 =================
def query_history(error_value, machine_id="Bearing_Motor_01"):
    """查询历史相似工单"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT timestamp, error_value, report FROM work_orders
                 WHERE machine_id = ? AND error_value BETWEEN ? AND ?
                 ORDER BY timestamp DESC LIMIT 3''',
              (machine_id, error_value - 0.05, error_value + 0.05))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return "无历史工单记录"
    return "\n".join([f"[{r[0]}] 误差{r[1]:.4f}: {r[2][:100]}..." for r in rows])

def save_work_order(error_value, report, machine_id="Bearing_Motor_01"):
    """保存工单到历史库"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO work_orders (timestamp, machine_id, error_value, report)
                 VALUES (?, ?, ?, ?)''',
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), machine_id, error_value, report))
    conn.commit()
    conn.close()
    print("📝 工单已写入历史库")

def local_rule_engine(error_value):
    """本地降级引擎"""
    return json.dumps({
        "故障类型": "轴承严重异常（本地规则判定）",
        "严重程度": "高危",
        "紧急建议": f"重构误差 {error_value:.4f}，超过阈值 0.15，建议立即停机检查并更换轴承。",
        "预计维修时间": "2-4 小时",
        "数据来源": "本地降级引擎"
    }, ensure_ascii=False, indent=4)

# ================= 6. Agent 核心调度 =================
def agent_pipeline(error_value, machine_id="Bearing_Motor_01"):
    print(f"\n🔔 系统报警：重构误差 {error_value:.4f}")

    if error_value < THRESHOLD:
        print("✅ 设备健康，无需干预。")
        return

    # 6.1 查询历史工单
    history_context = query_history(error_value, machine_id)
    print(f"📜 历史工单检索：{history_context[:80]}...")

    # 6.2 RAG 检索手册
    query = f"重构误差 {error_value:.4f}，严重异常，可能是什么故障？如何维修？"
    results = collection.query(query_texts=[query], n_results=2)
    manual_context = "\n\n".join(results['documents'][0])
    print(f"📚 已检索到 {len(results['documents'][0])} 条手册条款")

    # 6.3 构造 Prompt
    prompt = f"""
    你是一个工业设备维修专家。设备 {machine_id} 的轴承异常检测系统报警。
    重构误差为 {error_value:.4f}（正常阈值是 0.01）。

    【历史维修记录】：
    {history_context}

    【维修手册条款】：
    {manual_context}

    请严格依据以上信息，生成一份维修工单（JSON格式）。
    包含字段：故障类型、严重程度、紧急建议、预计维修时间。
    如果历史记录显示该设备近期多次报警，请在建议中特别提示。
    """

    # 6.4 调用云端大模型
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=15
        )
        report = response.choices[0].message.content
        mode = "云端 DeepSeek + RAG + 历史记忆"
    except Exception as e:
        print(f"⚠️ 云端失败，触发降级: {e}")
        report = local_rule_engine(error_value)
        mode = "本地降级规则引擎"

    print(f"\n✅ 报告生成完毕（模式：{mode}）")
    print("=" * 50)
    print(report)
    print("=" * 50)

    # 6.5 保存工单
    save_work_order(error_value, report, machine_id)

# ================= 7. 主程序入口 =================
if __name__ == "__main__":
    print("🚀 边缘端 Agent 系统启动...")
    dummy_signal = np.random.randn(1024) * 0.1 + np.sin(np.linspace(0, 50, 1024)) * 2.5
    error_val = detect_anomaly(dummy_signal)
    agent_pipeline(error_val, machine_id="Bearing_Motor_01")