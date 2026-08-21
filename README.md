---
title: PangPang RAG Chatbot
emoji: 🍞
colorFrom: yellow
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

# ปังปัง RAG Chatbot — Pivot จาก MilkLab (Session 3)

แชทบอทตอบคำถามเกี่ยวกับเมนูขนมปังปิ้งและร้าน "ปังปัง" โดยใช้ RAG (Retrieval-Augmented Generation)
ดึงข้อมูลจาก `menu_kb.md` แล้วสร้างคำตอบด้วย Gemini API

โครง RAG (chunk -> embed -> FAISS -> retrieve -> Gemini prompt) และ observability
(`traces.jsonl`) ทั้งหมดสืบทอดมาจากเทมเพลต MilkLab เดิมโดยไม่แก้ logic —
สิ่งที่ปรับสำหรับ pivot นี้คือ knowledge base (`menu_kb.md`) และ branding ใน UI
ดูรายละเอียดการ pivot ทั้งหมดใน `PIVOT.md`
