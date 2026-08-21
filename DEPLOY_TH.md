# 2.3 Deploy HuggingFace Spaces — คู่มือสั้น

## สิ่งที่ต้องมีในโฟลเดอร์ repo
```
app.py
menu_kb.md
requirements.txt
.github/workflows/deploy-to-hf.yml   <- มาจาก template repo ที่ให้มาแล้ว
```

## ขั้นตอน

1. **สร้าง Space บน HuggingFace**
   - ไปที่ huggingface.co → New Space → เลือก SDK เป็น **Streamlit**
   - ตั้งชื่อ Space ตาม convention ของ pivot: `<pivot-domain>-rag` เช่น `pangpang-rag`

2. **สร้าง HF Token**
   - HuggingFace → Settings → Access Tokens → New token (สิทธิ์ **write**)
   - คัดลอก token เก็บไว้

3. **ตั้ง secret `HF_TOKEN` ในทั้งสองที่**
   - **GitHub Codespaces**: repo Settings → Secrets and variables → Codespaces → New secret ชื่อ `HF_TOKEN`
   - **GitHub Actions**: repo Settings → Secrets and variables → Actions → New secret ชื่อ `HF_TOKEN`
   - (ถ้ามี secret สำหรับ Gemini ด้วย เช่น `GEMINI_API_KEY` ให้ตั้งเป็น Space secret ที่หน้า Settings ของ Space นั้นแทน เพราะ workflow ไม่ได้ส่งค่านี้เข้าไป)

4. **ตั้งค่า Gemini API key ที่ฝั่ง Space**
   - หน้า Space → Settings → Variables and secrets → New secret
     - name: `GEMINI_API_KEY`
     - value: API key จาก Google AI Studio

5. **Push โค้ดขึ้น branch `main`**
   ```
   git add app.py menu_kb.md requirements.txt
   git commit -m "Session 3: RAG chatbot"
   git push origin main
   ```
   GitHub Action (`deploy-to-hf.yml`) จะ trigger อัตโนมัติเมื่อ push เข้า `main`
   แล้ว push โค้ดต่อไปที่ HF Space ให้เอง

6. **รอ build เสร็จ**
   - ใช้เวลาประมาณ 2-3 นาที ดูสถานะได้ที่แท็บ **Actions** ของ GitHub repo
   - หรือดู log โดยตรงที่หน้า Space (แท็บ **Logs**)

## Troubleshooting ที่เจอบ่อย
- **`ModuleNotFoundError`**: เช็คว่า `requirements.txt` มีครบ (โดยเฉพาะ `faiss-cpu`, `sentence-transformers`)
- **Build ค้างนาน**: sentence-transformers โหลดโมเดลครั้งแรกช้า ปกติของ cold start
- **Gemini error / 403**: เช็คว่าใส่ `GEMINI_API_KEY` เป็น **Space secret** แล้ว (ไม่ใช่ GitHub secret)
- **traces.jsonl หายหลัง restart**: ปกติ เพราะ Space filesystem เป็นแบบชั่วคราว — ถ้าอยากเก็บถาวรให้ export ไปที่อื่น เช่น เขียนต่อท้าย Google Sheets ที่ต่อไว้แล้วใน Session 2
