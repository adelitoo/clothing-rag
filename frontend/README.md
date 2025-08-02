# Fashion Search Frontend

This is a Streamlit-based frontend for interacting with the fashion search backend. It allows users to input text queries and retrieve visually similar items using embeddings and a vector database.

---

## 🚀 Getting Started

Follow these steps to set up and run the frontend:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-frontend-repo.git
cd your-frontend-repo
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

- On **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```
- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Insert Embeddings into Milvus

Run the following command from the **root folder** to populate Milvus with embeddings:

```bash
python send_request.py --mode pipeline --insert-db
```

---

## 🖥️ Launch the Streamlit App

After embeddings are inserted, run:

```bash
streamlit run app.py
```

The app will be available at:

```
http://localhost:8501
```

---

## ✅ Summary

- Streamlit provides the user interface for submitting fashion queries.
- Embeddings are inserted into Milvus using `send_request.py`.
- Results are fetched via the backend and displayed visually.

---
