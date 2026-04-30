# Parent-Model Deployment

This folder is the deployment target for the original FastAPI parent-model service.

## Runtime files

- `api.py`
- `requirements.txt`
- `Dockerfile`
- `.env.example`

## Run locally

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r "parent_model\requirements.txt"
$env:API_KEY="change-me"
& ".\.venv\Scripts\python.exe" -m uvicorn parent_model.api:app --host 0.0.0.0 --port 8000
```

## Docker

```powershell
docker build -f "parent_model/Dockerfile" -t usl-disease-api .
docker run --rm -p 8000:8000 --env-file "parent_model/.env.example" usl-disease-api
```

## Separation

- The parent-model deployment lives here in `parent_model`.
- The child-model deployment lives in `child_model`.
- Keep them deployed separately so each target can have its own runtime and artifact settings.
