# Backend (FastAPI)

## Run

From `backend/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Configuration is loaded from a `.env` file. For local development, the simplest approach is to place `.env` in the **repository root** (one level above `backend/`) *or* copy the same variables into `backend/.env`.

Environment variables are loaded from (in order) repo `../.env`, `backend/.env`, then `backend/.env.local`. **Empty values in a later file do not erase** values from an earlier file, so you can keep placeholder lines in `.env.local` only for keys you actually want to override.

If you copy real secrets into `.env.local`, fill every line you need; lines like `SHOPIFY_ACCESS_TOKEN=` with nothing after `=` are fine and will no longer wipe `backend/.env`.

## Key files

- `app/main.py`: FastAPI routes (`/api/chat`, `/health`)
- `app/agent_service.py`: Gemini tool-calling agent loop + system prompt
- `app/shopify.py`: GET-only Shopify client + tools (`get_shopify_data`, guarded `python_repl`)
