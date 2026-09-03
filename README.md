# 🏺 ศูนย์วิจัยขนมไทยดิจิทัล ตำบลจอมทอง (AI RAG Assistant)

ระบบผู้ช่วย AI ให้ข้อมูลอาหารพื้นถิ่น วัฒนธรรม และภูมิปัญญาท้องถิ่น ตำบลจอมทอง จังหวัดพิษณุโลก โดยใช้เทคโนโลยี **Retrieval-Augmented Generation (RAG)** ร่วมกับ **OpenAI GPT-4o-mini**, **FAISS Vector Store** และ **Streamlit**

---

## 🌟 ฟีเจอร์หลัก (Key Features)

- **Semantic Search**: ค้นหาข้อมูลเชิงความหมายจากฐานความรู้เอกสาร (`data.txt`)
- **Interactive Discovery Menu**: เมนูอาหารพื้นถิ่น 20 รายการและคำถามที่พบบ่อย (Quick Prompts) คลิกเพื่อถาม AI ได้ทันที
- **Glassmorphism UI**: หน้าจอการใช้งานดีไซน์ทันสมัย รองรับทั้ง Desktop, Tablet และ Mobile (Responsive)
- **Strict Knowledge Boundaries**: ตอบเฉพาะข้อมูลอ้างอิงที่มีในฐานความรู้เพื่อป้องกันการหลอนของ AI (Hallucination)

---

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)

- **Frontend / Framework**: Streamlit
- **LLM / Embedding**: OpenAI (`gpt-4o-mini`, `text-embedding-ada-002`)
- **Orchestration**: LangChain (`langchain-openai`, `langchain-community`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)

---

## 🚀 วิธีการติดตั้งและใช้งาน (Installation & Setup)

### 1. Clone Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. สร้างและเปิดใช้งาน Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 4. ตั้งค่า Environment Variables
คัดลอกไฟล์ `.env.example` เป็น `.env` และกรอก API Key ของ OpenAI:
```bash
# Windows (PowerShell)
cp .env.example .env

# หรือสร้างไฟล์ .env และใส่รหัส
OPENAI_API_KEY=sk-proj-your-actual-api-key
```

### 5. เริ่มใช้งานแอปพลิเคชัน
```bash
streamlit run app.py
```

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
├── app.py              # แอปพลิเคชัน Streamlit หลัก (UI + RAG Chain)
├── program.py          # Script ตัวอย่างทดสอบ RAG Pipeline
├── data.txt            # ฐานข้อมูลคลังความรู้เกี่ยวกับอาหารพื้นถิ่น
├── requirements.txt    # รายชื่อ Library ที่ต้องติดตั้ง
├── .env.example        # ตัวอย่างไฟล์คอนฟิก Environment Variable
├── .gitignore          # ซ่อนไฟล์ที่ไม่ต้องการ Push ขึ้น GitHub (.env, .venv)
└── README.md           # เอกสารอธิบายโปรเจกต์
```
