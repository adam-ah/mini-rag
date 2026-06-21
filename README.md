# Mini RAG

Mini RAG is a lightweight Retrieval-Augmented Generation system that converts a variety of documents (PDFs, Word, etc.) into a searchable text corpus. It allows you to ask questions and receive grounded, synthesized answers with citations, using adaptive refinement for complex queries.

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure LLM**:
   Copy `.env.example` to `.env` and set your model and endpoint. Using a local LLM (e.g., via Ollama or LM Studio) is highly encouraged to get the best summaries and maintain privacy.

3. **Launch**:
   ```bash
   ./start.sh
   ```
   Then open `http://127.0.0.1:5000` in your browser.
