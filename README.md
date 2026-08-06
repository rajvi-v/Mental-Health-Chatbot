# Student Support Chat

A student-support chatbot with a plain HTML/CSS/JavaScript interface and a FastAPI RAG backend. LangChain loads and chunks two local PDFs, OpenAI `text-embedding-3-large` creates embeddings, Chroma stores the vectors, and GPT-4o mini writes concise answers using only retrieved document context.

Messages remain only in the current browser tab and are not stored.

## Knowledge base

The application indexes:

- data/knowledge_base/How-to-help-a-student-in-crisis-flowchart-2022-2023.pdf
- data/knowledge_base/student-placement-year-handbook.pdf
- data/knowledge_base/List of emotions-reasons-services Interns 16Aug2021.xlsx

The workbook powers both typed wellbeing questions and the feelings picker in the sidebar. The persistent Chroma index is stored in var/chroma/ and is ignored by Git. A document hash manifest causes the index to rebuild automatically when any indexed file or indexing setting changes.

## Prerequisites

- Python 3.11 or newer
- An OpenAI API key for embeddings and answer generation
- An internet connection so the backend can call OpenAI

OpenAI is used to create searchable embeddings and to turn retrieved PDF passages into concise answers.

## Configuration

Create .env in the project root:

    OPENAI_API_KEY=your_key_here
    OPENAI_MODEL=gpt-4o-mini
    OPENAI_TIMEOUT_SECONDS=30
    RAG_MIN_RELEVANCE=0.35

Never put the API key in app/static/app.js or commit .env.

## Run locally

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m uvicorn app.main:app --reload

Open http://127.0.0.1:8000. If that port is unavailable, add --port 8001. The sidebar includes a feelings picker once the backend has loaded the emotions workbook.

The first startup can take longer while the Chroma collection is created. Later startups reuse the persisted collection when the PDFs and indexing settings are unchanged.

## Chat behaviour

- Standalone greetings such as Hi receive a short fixed response without retrieval or an OpenAI request.
- Relevant questions retrieve up to three PDF chunks and receive a grounded 2-4 sentence answer.
- Weak matches return a short knowledge-base fallback without calling OpenAI.
- Short follow-up questions use the latest user message as retrieval context.
- Crisis language still displays the fixed UK urgent-support warning.

## Tests

    python -m pip install -r requirements.txt
    python -m pytest

The automated tests replace Chroma, embeddings, and OpenAI with test doubles, so they do not use API credits.



