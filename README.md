```markdown
# 🧵 Full-Stack Fashion Search

This project is a complete fashion search application that uses natural language queries to find visually similar items. It combines a **FastAPI** backend for search logic with a **Streamlit** frontend for user interaction.

The core technologies include:
* **Ollama (Llama 3.1):** Rewrites and enhances user search queries.
* **CLIP:** Generates text and image embeddings.
* **Milvus:** A vector database for storing and searching embeddings at scale.
* **FastAPI:** Serves the backend search API.
* **Streamlit:** Provides the interactive web user interface.

## 📁 Project Structure

```

.
├── backend/
│   ├── main.py
│   ├── scripts/
│   └── ...
├── frontend/
│   ├── app.py
│   ├── send\_request.py
│   └── ...
└── README.md         \<-- You are here

````

---

## 🚀 Getting Started (End-to-End Setup)

Follow these steps in order to set up and run the entire application.

### Prerequisites

Make sure you have the following installed on your system:
* **Git**
* **Python 3.9+**
* **Docker** and **Docker Compose**
* **Ollama:** Download and install from [https://ollama.com](https://ollama.com)

### Step 1: Set Up Ollama

Before launching the backend, you need to pull the required Large Language Model using Ollama.

1.  Start the Ollama server (this may happen automatically after installation). If not, run:
    ```bash
    ollama serve
    ```

2.  In a new terminal, pull the Llama 3.1 model (this only needs to be done once):
    ```bash
    ollama pull llama3.1:8b
    ```

---

### Step 2: Launch the Backend Server

The backend runs Milvus and the FastAPI application. We'll use the provided script for a quick setup.

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```

2.  Give the startup script execution permissions (only needs to be done once):
    ```bash
    chmod +x scripts/start_backend.sh
    ```

3.  Run the script. The first time, use `source` to ensure the virtual environment is activated correctly for your session:
    ```bash
    source scripts/start_backend.sh
    ```
    > For subsequent runs, you can just execute `./scripts/start_backend.sh`.

This script will automatically start the Milvus Docker container, install Python dependencies into a `.venv`, and launch the FastAPI server.

**Leave this terminal running.** The backend is now available at `http://localhost:8000`.

---

### Step 3: Populate the Vector Database

Next, you need to insert the fashion item embeddings into Milvus. This is done using a script in the `frontend` directory.

1.  **Open a new terminal**.

2.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```

3.  Set up the Python virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4.  Run the ingestion script to populate Milvus:
    ```bash
    python send_request.py --mode pipeline
    ```
    This may take a few moments. Once it's complete, the database is ready.

---

### Step 4: Launch the Frontend UI

Finally, launch the Streamlit web application.

1.  In the **same frontend terminal** (where the `venv` is active), run:
    ```bash
    streamlit run app.py
    ```

2.  Open your web browser and navigate to:
    **[http://localhost:8501](http://localhost:8501)**

You should now see the fashion search interface and be able to submit queries! ✨
````
