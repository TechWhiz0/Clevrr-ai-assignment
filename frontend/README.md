# Frontend (React + Vite)

## Run

```powershell
cd frontend
npm install
npm run dev
```

## API base URL

By default the Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

To call a remote backend, set:

```text
VITE_API_BASE=https://your-api-host.example
```
