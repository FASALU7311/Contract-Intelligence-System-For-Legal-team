# 🗐 ℂ𝕠𝕟𝕥𝕣𝕒𝕔𝕥 𝕀𝕟𝕥𝕖𝕝𝕝𝕚𝕘𝕖𝕟𝕔𝕖 𝕊𝕪𝕤𝕥𝕖𝕞 𝕗𝕠𝕣 𝕃𝕖𝕘𝕒𝕝 𝕋𝕖𝕒𝕞𝕤
<a href="https://github.com/404"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></a>

An AI-powered system that helps **legal teams** analyze contracts faster and more effectively.  
It classifies clauses, extracts deadlines & risks, generates summaries, and provides actionable insights.  

<img align="right" alt="Coding" width="300" src="https://i.pinimg.com/originals/81/17/8b/81178b47a8598f0c81c4799f2cdd4057.gif">
---

## ✦ Features  
- ▣ **Clause Classification** – Detects contract clauses using **Legal-BERT**.  
- ⏱ **Deadline Extraction** – Finds important dates & timelines.  
- ⚠︎ **Risk Detection** – Highlights potential legal & financial risks.  
- ✎ **Summarization** – Generates easy-to-read summaries with **PEGASUS**.  
- ▤ **Insights Dashboard** – Interactive Streamlit app for analysis.  
- ⬎ **Export Reports** – Download results in **PDF, DOCX, CSV, TXT, XLSX** formats.  
- ⌑ **User Authentication** – Register, login, OTP verification & password reset.  

---

## 🏗️ Tech Stack
- **Python 3.12+**  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- **Legal-BERT** (`nlpaueb/legal-bert-base-uncased`)  
- **PEGASUS-XSUM** for summarization  
- **Streamlit** for frontend UI  
- **PyMuPDF & python-docx** for document parsing  
- **Pandas, Scikit-learn** for data handling  

---

## 📂 Project Structure

```
contract-intelligence-system/
├── app/
│   ├── streamlit_app.py          # 🎯 Main Streamlit frontend app
│   ├── utils.py                  # ⚙️ Helper functions (text extraction, clause splitting, etc.)
│   ├── auth.py                   # 🔐 Authentication (register, login, OTP, password reset)
│   ├── analysis.py               # 🤖 Clause classification, summarization, insights
│   ├── download.py               # 📥 Export reports (PDF, DOCX, CSV, XLSX, TXT)
│   └── requirements.txt          # 📦 Python dependencies
│
├── model/
│   ├── train_model.py            # 🏋️ Fine-tuning Legal-BERT on CUAD dataset
│   ├── legalbert_model/          # 📂 Saved trained Legal-BERT model + tokenizer
│   ├── label_mapping.csv         # 🗂 Mapping between clause ID ↔ clause type
│   └── logs/                     # 📝 Training logs & checkpoints
│
├── data/
│   ├── cuad_final.csv            # 📊 CUAD dataset (training data)
│   ├── sample_contracts/         # 📂 Sample contracts in PDF, DOCX, TXT
│   └── predicted_clauses.csv     # ✅ Predicted results after analysis
│
├── assets/
│   ├── demo.gif                  # 🎥 Demo animation for README
│   ├── diagram.png               # 📊 Project flow diagram
│   └── logo.png                  # 🖼 Logo or icon for the system
│
├── users.csv                     # 👤 User authentication database
├── README.md                     # 📘 Project overview + setup instructions
├── LICENSE                       # ⚖️ License file (MIT, Apache, etc.)
└── .gitignore                    # 🚫 Ignore unnecessary files (logs, checkpoints, etc.)

```

## 📊 Demo (Screenshots)

- Login & Registration with OTP
- Upload Contract (PDF/DOCX/CSV)
- Clause Analysis (Type, Risks, Deadlines)
- Summaries & Insights Dashboard
- Download Reports in Multiple Formats

## 🚀 Future Improvements

- Add support for multi-language contracts.
- Improve accuracy with fine-tuned transformer models.
- Integrate with legal knowledge bases.

<a href="https://github.com/404"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></a>

## 🤝 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

<a href="https://github.com/404"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></a>
<a href="https://github.com/404"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></a>

