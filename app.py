"""
Session 3: Demi RAG Chatbot + Eval
==================================
ทำตามข้อ 2.2 (5 ข้อ) + TODO 6 (observability):
  1. โหลด menu_kb.md แล้ว split เป็น chunk
  2. encode chunk ด้วย sentence-transformers (multilingual-MiniLM)
  3. สร้าง faiss index จาก embedding cache ด้วย st.cache_resource
  4. สร้าง chat UI ด้วย st.chat_input + st.chat_message
  5. retrieve top-k chunk สำหรับคำถาม แล้วใส่ใน Gemini prompt
  6. (TODO ของ observability) ห่อ generate_answer ด้วย span, ใช้ trace_id เดียวกันกับ
     retrieve_top_k, log ลง traces.jsonl

รัน: streamlit run app.py
ต้องตั้งค่า environment variable GEMINI_API_KEY ก่อนรัน (หรือใส่ใน .streamlit/secrets.toml)
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from google import genai

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
KB_PATH = Path(__file__).parent / "menu_kb.md"
TRACES_PATH = Path(__file__).parent / "traces.jsonl"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-chatbot")

# Gemini setup — ใช้ st.secrets ก่อน แล้ว fallback ไป env var
# หมายเหตุ: ไลบรารีเก่า google-generativeai ถูกเลิกใช้แล้ว (deprecated 31 ส.ค. 2025)
# และไม่รองรับโมเดลรุ่นใหม่ (Gemini 3.x) เลย ต้องใช้ SDK ใหม่ google-genai แทน
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
GEMINI_MODEL_NAME = "gemini-3.5-flash"
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ----------------------------------------------------------------------------
# ข้อ 1: โหลด menu_kb.md แล้ว split เป็น chunk
# ----------------------------------------------------------------------------
def load_and_chunk(path: Path) -> list[str]:
    """แบ่งไฟล์ markdown เป็น chunk ตามหัวข้อ (## หรือ ###) เพื่อให้แต่ละ chunk
    มีข้อมูลครบเรื่องเดียว (เช่น 1 เมนู หรือ 1 คำถาม FAQ)

    หมายเหตุ: ตัด preamble ก่อนหัวข้อแรก (เช่น หัวเรื่อง H1 บนสุดของไฟล์) ทิ้งไป
    เพราะไม่ใช่ chunk ที่มีเนื้อหาจริง — เดิมโค้ดจุดนี้มีบั๊กทำให้เกิด chunk ปลอม
    ที่ index 0 จนตัวเลข chunk index ทั้งหมดเพี้ยนไปเทียบกับ ground truth ใน eval.ipynb"""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith("## ") or l.startswith("### ")), 0)
    lines = lines[start:]

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if current:
                joined = "\n".join(current).strip()
                if len(joined) > 10:
                    chunks.append(joined)
            current = [line]
        else:
            current.append(line)
    if current:
        joined = "\n".join(current).strip()
        if len(joined) > 10:
            chunks.append(joined)
    return chunks


# ----------------------------------------------------------------------------
# ข้อ 2+3: encode chunk ด้วย sentence-transformers แล้วสร้าง faiss index
# cache ด้วย st.cache_resource เพื่อไม่ต้อง encode ใหม่ทุกครั้งที่ rerun
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังเตรียม knowledge base...")
def build_index():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    chunks = load_and_chunk(KB_PATH)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product บน normalized vectors = cosine similarity
    index.add(embeddings)
    return model, index, chunks


# ----------------------------------------------------------------------------
# ข้อ 5 (ส่วน retrieval): retrieve top-k chunk สำหรับคำถาม
# ----------------------------------------------------------------------------
def retrieve_top_k(query: str, model, index, chunks, k: int = TOP_K, trace_id: str = None):
    """คืนค่า list ของ (chunk_text, score) เรียงจากใกล้เคียงที่สุด
    log ผลลัพธ์ลง traces.jsonl โดยใช้ trace_id เดียวกันกับ generate_answer"""
    t0 = time.time()
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_vec, k)
    results = [(chunks[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i != -1]

    _log_span(
        trace_id=trace_id,
        span_name="retrieve_top_k",
        input_data={"query": query, "k": k},
        output_data={
            "results": [{"chunk": c[:200], "score": s} for c, s in results],
        },
        duration_ms=(time.time() - t0) * 1000,
    )
    return results


# ----------------------------------------------------------------------------
# ข้อ 5 (ส่วน generation) + TODO 6: ห่อ generate_answer ด้วย span, ใช้ trace_id
# เดียวกันกับ retrieve_top_k, log ลง traces.jsonl
# ----------------------------------------------------------------------------
def generate_answer(query: str, retrieved: list[tuple[str, float]], trace_id: str = None) -> str:
    t0 = time.time()
    context = "\n\n---\n\n".join(chunk for chunk, _ in retrieved)

    prompt = f"""คุณคือแชทบอทตอบคำถามลูกค้าเกี่ยวกับเมนูและร้าน โดยใช้ข้อมูลด้านล่างเท่านั้น
ถ้าไม่มีข้อมูลที่เกี่ยวข้อง ให้บอกว่าไม่ทราบ ห้ามเดาข้อมูลเอง

กติกาการตอบเรื่องเมนู:
- เมื่อลูกค้าถามถึงเมนูใดเมนูหนึ่ง ให้บอกส่วนผสม (ingredients) ของเมนูนั้นประกอบคำตอบเสมอ ถ้าข้อมูลอ้างอิงมีระบุไว้
- ถ้าเมนูนั้นมีข้อมูล Allergen ระบุไว้ ให้เพิ่มคำเตือนต่อท้ายคำตอบเสมอ ในรูปแบบ
  "⚠️ เมนูนี้มีส่วนผสมของ [Allergen] ผู้ที่แพ้ [Allergen] ควรหลีกเลี่ยงหรือแจ้งพนักงานก่อนสั่ง"
- ถ้าลูกค้าถามหาเมนูที่ไม่มีสารก่อภูมิแพ้บางอย่าง (เช่น ไม่ใส่นม) ให้แนะนำเฉพาะเมนูที่ข้อมูลอ้างอิงยืนยันว่าไม่มี allergen นั้นจริง ๆ
- ห้ามสรุปเอาเองว่าเมนูใดปลอดภัยหรือไม่มี allergen ถ้าข้อมูลอ้างอิงไม่ได้ระบุไว้ชัดเจน ให้บอกว่าไม่มีข้อมูลและแนะนำให้สอบถามพนักงานแทน

ข้อมูลอ้างอิง:
{context}

คำถามลูกค้า: {query}

คำตอบ:"""

    try:
        if gemini_client is None:
            raise RuntimeError("ยังไม่ได้ตั้งค่า GEMINI_API_KEY")
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        answer = response.text
    except Exception as e:
        logger.exception("Gemini generation failed")
        answer = f"เกิดข้อผิดพลาดในการเรียก Gemini: {e}"

    _log_span(
        trace_id=trace_id,
        span_name="generate_answer",
        input_data={"query": query, "prompt_chars": len(prompt)},
        output_data={"answer": answer},
        duration_ms=(time.time() - t0) * 1000,
    )
    return answer


# ----------------------------------------------------------------------------
# Observability helper — เขียน span ทุกครั้งลง traces.jsonl (append)
# หมายเหตุ (จาก lab): บน HuggingFace Space ไฟล์นี้จะหายเมื่อ restart เพราะ
# filesystem เป็นแบบชั่วคราว — ถ้าต้องเก็บถาวร ให้ส่งออกไปที่อื่น เช่น
# เขียนต่อท้าย Google Sheets (ตามที่ต่อไว้แล้วใน S2)
# ----------------------------------------------------------------------------
def _log_span(trace_id: str, span_name: str, input_data: dict, output_data: dict, duration_ms: float):
    record = {
        "trace_id": trace_id,
        "span": span_name,
        "timestamp": time.time(),
        "duration_ms": round(duration_ms, 1),
        "input": input_data,
        "output": output_data,
    }
    logger.info("trace=%s span=%s duration=%.1fms", trace_id, span_name, duration_ms)
    with open(TRACES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_trace_for_id(trace_id: str) -> list[dict]:
    """อ่าน span ทั้งหมดที่มี trace_id เดียวกัน สำหรับแสดงใน expander 'Trace'"""
    if not TRACES_PATH.exists():
        return []
    spans = []
    with open(TRACES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("trace_id") == trace_id:
                    spans.append(rec)
            except json.JSONDecodeError:
                continue
    return spans


# ----------------------------------------------------------------------------
# ข้อ 4: Streamlit chat UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Demi RAG Chatbot", page_icon="💬")
    st.title("💬 Demi RAG Chatbot")
    st.caption("ถามคำถามเกี่ยวกับเมนูและร้านได้เลย — ตอบจาก knowledge base เท่านั้น")

    if not GEMINI_API_KEY:
        st.warning("ยังไม่ได้ตั้งค่า GEMINI_API_KEY — ใส่ใน .streamlit/secrets.toml หรือ environment variable")

    model, index, chunks = build_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # แสดงประวัติแชท
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "trace_id" in msg:
                with st.expander("Trace"):
                    st.json(get_trace_for_id(msg["trace_id"]))

    # รับคำถามใหม่
    query = st.chat_input("พิมพ์คำถามของคุณ...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        trace_id = str(uuid.uuid4())

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาและตอบคำถาม..."):
                retrieved = retrieve_top_k(query, model, index, chunks, k=TOP_K, trace_id=trace_id)
                answer = generate_answer(query, retrieved, trace_id=trace_id)
            st.markdown(answer)
            with st.expander("Trace"):
                st.json(get_trace_for_id(trace_id))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "trace_id": trace_id}
        )


if __name__ == "__main__":
    main()
