# LLM Chatbot (RAG-Powered)

## Backend setup
1. Create and activate virtual env:

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:

   ```bash
   uvicorn app.main:app --reload
   ```