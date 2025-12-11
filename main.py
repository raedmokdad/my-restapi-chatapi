# main.py
import os
import re
import json
import math
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi import Header
import os
from dotenv import load_dotenv 
from pydantic import BaseModel, Field
from typing import Any, Dict
from pathlib import Path


# Load environment variables from .env file if it exists
load_dotenv()

# Config from env
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
API_PASSWORD = os.getenv("API_PASSWORD")

# Where templates are stored (adjust if your files are elsewhere)
PROMPT1_PATH = os.getenv("PROMPT1_PATH", "prompt1.txt")
PROMPT2_PATH = os.getenv("PROMPT2_PATH", "prompt2.txt")
PROMPT3_PATH = os.getenv("PROMPT3_PATH", "prompt3.txt")
PROMPT4_PATH = os.getenv("PROMPT4_PATH", "prompt4.txt")
MESSTYPE_PATH = os.getenv("MESSTYPE_PATH", "messagetype.txt")


# Use env var so Railway mount path can be configured; fallback to local "jsons" for dev.
JSONS_DIR = Path(os.environ.get("JSONS_DIR", "jsons"))
JSONS_DIR.mkdir(parents=True, exist_ok=True)

# Allow letters, numbers, hyphen, underscore, and dot
FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

class CreateJsonRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Filename (without path)")
    proplist: Dict[str, Any] = Field(..., description="Arbitrary JSON object to store")

def sanitize_filename(name: str) -> str:
    """
    Sanitize the filename to allow only safe characters.
    If the name contains unsafe characters, replace them with underscores.
    Trims leading/trailing dots, underscores, hyphens.
    Limits length to 100 characters.
    """
    if not FILENAME_RE.match(name):
        cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        cleaned = cleaned.strip("._-")
        if not cleaned:
            raise ValueError("Invalid file name after sanitization.")
        return cleaned[:100]
    return name

def filepath_for(name: str) -> Path:
    safe = sanitize_filename(name)
    return JSONS_DIR / f"{safe}.json"

def load_json(name: str) -> dict:
    path = filepath_for(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if not (AZURE_ENDPOINT and AZURE_API_KEY):
    # For safety: app will start but will reject requests if keys missing
    print("Warning: AZURE_ENDPOINT or AZURE_API_KEY not set. Set them as environment variables.")

app = FastAPI(title="Car Buyer Message API")

# Load templates at startup
def _load_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

PROMPT_TEMPLATES = {
    "person1": _load_file(PROMPT1_PATH),
    "person2": _load_file(PROMPT2_PATH),
    "person3": _load_file(PROMPT3_PATH),
    "person4": _load_file(PROMPT4_PATH),
}
SYSTEM_TEMPLATE = _load_file(MESSTYPE_PATH)


class CarInfo(BaseModel):
    seller: str
    marke: str
    modell: str
    ez: Optional[str] = None
    getriebe: Optional[str] = None
    fuel: Optional[str] = None
    km: Optional[str] = None
    aufbau: Optional[str] = None
    preis: Optional[str] = None
    beschreibung: Optional[str] = ""
    telefon: Optional[str] = ""
    buyer: Optional[str] = ""
    person_type: str  # "person1" or "person2" (decides which prompt to use)



async def verify_password(password: str = Header(None)):
    if API_PASSWORD is None:
        raise HTTPException(500, "API password not set on server.")
    if password != API_PASSWORD:
        raise HTTPException(401, "Unauthorized")

def fill_placeholders(template: str, fields: dict) -> str:
    """
    Replace placeholders like {seller}, {marke}, {modell}, {preisvorschlag}, etc.
    Template files appear to use these placeholders.
    """
    def safe_replace(match):
        key = match.group(1)
        return str(fields.get(key, match.group(0)))

    # replace {key} placeholders
    filled = re.sub(r"\{(\w+)\}", safe_replace, template)
    return filled


async def call_azure_chat(messages: list) -> str:
    """
    Calls the Azure OpenAI chat/completions endpoint (synchronous via async httpx).
    Returns assistant message content.
    """
    if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_DEPLOYMENT):
        raise HTTPException(status_code=500, detail="Azure OpenAI config missing (set AZURE_ENDPOINT, AZURE_API_KEY, AZURE_DEPLOYMENT).")

    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {
        "api-key": AZURE_API_KEY,
        "Content-Type": "application/json",
    }

    body = {
        "messages": messages,
        # you can expose temperature/max_tokens via env or keep fixed:
        "temperature": 0.3,
        "max_tokens": 40,
        "frequency_penalty": 0.2,   
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            # forward error text for easier debugging
            raise HTTPException(status_code=502, detail=f"Azure OpenAI error: {resp.status_code} {resp.text}")
        j = resp.json()
        # safe navigation: choices[0].message.content
        try:
            return j["choices"][0]["message"]["content"]
        except Exception:
            raise HTTPException(status_code=502, detail=f"Unexpected Azure response format: {j}")


@app.post("/generate-message")
async def generate_message(
    car: CarInfo,
    password: str = Header(None)
):
    await verify_password(password)
    # 1) choose template
    prompt_template = PROMPT_TEMPLATES.get(car.person_type)
    if not prompt_template:
        raise HTTPException(status_code=400, detail=f"Unknown person_type '{car.person_type}'. Valid: {list(PROMPT_TEMPLATES.keys())}")


    # 2) prepare fields dictionary for replacement
    fields = {
        "seller": car.seller,
        "marke": car.marke,
        "modell": car.modell,
        "ez": car.ez or "",
        "getriebe": car.getriebe or "",
        "fuel": car.fuel or "",
        "km": car.km or "",
        "aufbau": car.aufbau or "",
        "preis": car.preis or "",
        "beschreibung": car.beschreibung or "",
        "telefon": car.telefon or "",
        "buyer": car.buyer or "",
        "preisvorschlag": "",
    }

    # 3) fill prompt and system
    user_prompt = fill_placeholders(prompt_template, fields)
    system_prompt = fill_placeholders(SYSTEM_TEMPLATE, fields) if SYSTEM_TEMPLATE else "You are a helpful assistant."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 4) call azure openai
    assistant_content = await call_azure_chat(messages)

    # 5) return result (raw assistant text)
    return {"message": assistant_content}


@app.post("/create-json", status_code=201)
async def create_json(payload: CreateJsonRequest):
    try:
        path = filepath_for(payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # atomic-ish write: write to temp file then rename
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload.proplist, f, ensure_ascii=False, indent=2)
            f.flush()
        tmp.replace(path)  # overwrite existing file if exists
    except Exception as e:
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to write file: {e}")

    return {
        "message": "created/updated",
        "filename": str(path)
    }



@app.get("/list-jsons")
async def list_jsons():
    files = [p.name for p in JSONS_DIR.glob("*.json")]
    return {"files": files}

@app.get("/read-json/{name}")
async def read_json(name: str):
    try:
        data = load_json(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"name": name, "proplist": data}

async def delete_json(name: str = Path(..., description="Name of the JSON file to delete")):
    """
    Delete a JSON file by name.
    """
    try:
        path = filepath_for(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File '{path.name}' not found.")

    try:
        path.unlink()  # delete the file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

    return {"message": "deleted", "filename": str(path)}