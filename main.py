# main.py
import os
import re
import json
import math
from typing import List, Optional
import httpx
from fastapi import Header
import os
from dotenv import load_dotenv 
from pydantic import BaseModel, Field
from typing import Any, Dict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Path as FastAPIPath
from fastapi import UploadFile, File, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi import Form
import string
import tempfile


# Load environment variables from .env file if it exists
load_dotenv()

# Config from env
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
API_PASSWORD = os.getenv("API_PASSWORD")


# Use env var so Railway mount path can be configured; fallback to local "jsons" for dev.
JSONS_DIR = Path(os.environ.get("JSONS_DIR", "jsons"))
JSONS_DIR.mkdir(parents=True, exist_ok=True)

# Allow letters, numbers, hyphen, underscore, and dot
FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_ATTEMPTS = 3

class CreateJsonRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Filename (without path)")
    proplist: Dict[str, Any] = Field(..., description="Arbitrary JSON object to store")


class UploadPromptRequest(BaseModel):
    name: str
    content: str

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
    """Get the full file path for a given JSON name after sanitization."""
    safe = sanitize_filename(name)
    return JSONS_DIR / f"{safe}.json"


def load_json(name: str) -> dict:
    """Load JSON file by name.
    Raises FileNotFoundError if not found.
    """
    path = filepath_for(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def load_prompt(prompt_name: str) -> str | None:
    """
    Load the content of a prompt text file by name.
    
    Args:
        prompt_name: Name of the prompt (without .txt)
    
    Returns:
        The content of the prompt as a string, or None if file doesn't exist.
    """
    file_path = JSONS_DIR / f"{prompt_name}.txt"
    
    if not file_path.exists():
        print(f"Prompt '{prompt_name}' not found.")
        return None
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading prompt '{prompt_name}': {e}")
        return None    




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
    max_tokens: Optional[int] = 40  # max tokens for response



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


async def call_azure_chat(messages: list, max_tokens: int) -> str:
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
        "max_tokens": max_tokens,
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

def get_attr(obj, attr, default=None):
    """Helper to get attribute or dict key safely."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)

def clean_word(word):
    """Remove punctuation from start and end of a word."""
    return word.strip(string.punctuation)

def validate_message(assistant_content: str, car, Greetinglist: List[str], needed_features: List[str], forbidden_phrases: List[str], max_tokens: Optional[int] = None) -> List[str]:
    
    """Validate the assistant message content.
    Returns a list of error messages. Empty list means validation passed.
    Validation checks:
    1. Length validation: message should not exceed max_tokens (if provided).
    2. Format-check: message starts with a greeting from Greetinglist followed by seller's name.
    3. Feature check: all needed_features are mentioned in the message.
    4. Price logic check: price strategy applied correctly.
    5. Blacklist filter: no forbidden phrases used.
    """
    
    errors = []

    # 1. Length validation
    if max_tokens is not None and len(assistant_content.split()) > max_tokens:
        errors.append("Message is too long to max_tokens")

    # 2. Format-check: starts with Greeting + Name
    first_word = clean_word(assistant_content.split()[0]) if assistant_content else ""
    second_word = clean_word(assistant_content.split()[1]) if len(assistant_content.split()) > 1 else ""
    seller_name = get_attr(car, "seller", "")
    if first_word not in Greetinglist or second_word != seller_name:
        errors.append(f"Message does not start with proper greeting and name ('first_word={first_word}' second_word={second_word} seller_name={seller_name}')")

    # 3. Feature check: needed features mentioned
    for feature in needed_features:
        feature_value = str(get_attr(car, feature, "")).lower()
        if feature_value not in assistant_content.lower():
            errors.append(f"{feature.capitalize()} not mentioned in message")

    # 4. Price logic check (example strategy)
    price = get_attr(car, "preis", None)
    if price:
        price = float(price)
        # Example: price <1000 -> double; 1000-3000 +50%, etc.
        expected_offer = None
        if price < 1000:
            expected_offer = price * 2
        elif 1000 <= price <= 3000:
            expected_offer = price * 1.5
        elif 3000 < price <= 10000:
            expected_offer = price * 1.2
        elif price > 10000:
            expected_offer = price * 1.1

        # Check if expected offer appears in the assistant content
        if expected_offer and str(round(expected_offer)) not in assistant_content:
            errors.append(f"Price strategy not applied correctly (expected offer: {round(expected_offer)})")

    # 5. Blacklist filter
    for phrase in forbidden_phrases:
        if phrase.lower() in assistant_content.lower():
            errors.append(f"Forbidden phrase used: '{phrase}'")

    return errors

# helper to build the corrective prompt the model will receive
def build_correction_prompt(original_message: str, validation_errors: list, greeting_list, features, blacklist, maxtoken: int, seller_name: str, preis_exists: bool):
    errors_text = "\n".join(f"- {e}" for e in validation_errors)
    # be explicit and prescriptive — ask the model to only return the corrected sentence
    prompt = f"""You returned this single-sentence message which failed validation:
ORIGINAL: {original_message}

Validation errors:
{errors_text}

Correct the message so it follows ALL the hard rules:
- Output EXACTLY one single sentence and nothing else.
- Must not exceed {maxtoken} tokens.
- Start with a greeting selected ONLY from {greeting_list} followed immediately by {seller_name} (no punctuation between greeting and name).
- Include these features: {features} naturally in the same sentence.
- Include a short personal reason for wanting the car.
- Pricing: {'If a numeric price exists, calculate the offer and include it in euros (rounded) according to strategy. Do NOT ask the price.' if preis_exists else 'If no price exists, ask "what price were you thinking?" naturally.'}
- DO NOT use any phrase from the blacklist: {blacklist}.
- No sign-offs, no extra explanation, no newlines, no emojis.

Return ONLY the corrected single sentence (no JSON, no commentary).
"""
    return prompt


def normalize_prices_in_text(text: str) -> str:
    """Normalize prices in text by removing formatting like commas, dots, spaces.
    E.g. "1,200.50" -> "1200.50"
    
    """
    def repl(match):
        number = match.group(0)
        return re.sub(r"[.,\s]", "", number)  # remove commas, dots, spaces
    return re.sub(r"\d[\d.,\s]*\d", repl, text)

def normalize_filename(name: str) -> str:
    """Normalize filename by stripping whitespace and replacing unsafe characters."""
    name = name.strip()
    name = re.sub(r"[^\w\-\.]", "_", name)
    return name

@app.post("/generate-message")
async def generate_message(
    car: CarInfo,
    password: str = Header(None)
):
    """
    Generate a buyer message based on car info and person type.
    Steps:
    0) verify password
    1) choose prompt template based on person_type
    2) prepare fields dictionary for replacement
    3) fill prompt and system
    4) call azure openai
    5) validate and possibly correct in a loop

    """
    await verify_password(password)
    # 1) choose template
    prompt_template = load_prompt(car.person_type) # PROMPT_TEMPLATES.get(car.person_type)
    SYSTEM_TEMPLATE = load_prompt("messagetype")

    if not prompt_template:
        raise HTTPException(status_code=400, detail=f"Unknown person_type '{car.person_type}'.")
    
    if not SYSTEM_TEMPLATE:
        raise HTTPException(status_code=400, detail=f"Unknown system prompt.")
    
    try:
        # Load JSON from volume
        prompt_json = load_json(car.person_type)  # uses JSONS_DIR

        # Safely get 'proplist'; fallback to root if 'proplist' missing
        data = prompt_json.get("proplist", prompt_json)

        # Extract lists with defaults
        greeting_list = data.get("Greetinglist", [])
        features = data.get("Features", [])
        blacklist = data.get("Blacklist", [])
        examples= data.get("Examples", [])

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

        # --- Convert Features ---
        filtered_features_dict = {key: fields.get(key, "") for key in features}
        filtered_features = ", ".join(f"{k}={v}" for k, v in filtered_features_dict.items() if v)
        # --- Convert Greetinglist ---
        greeting_list_str = ", ".join(greeting_list)
        black_list_str = ", ".join(blacklist)
        examples_list_str=  ", ".join(examples)

        # Ensure price is a number (float) if it exists
        preis_value = float(car.preis) if car.preis else None

        # 3) fill prompt and system
        prompt_parameters = {
        "Greetinglist": greeting_list_str,
        "Features": filtered_features,
        "Blacklist": black_list_str,
        "maxtoken": str(car.max_tokens),
        "seller": car.seller or "",
        "buyer": car.buyer or "",
        "preis": preis_value,
        "preisvorschlag": "",
        "Examples": examples_list_str
        }

        # merge all placeholders
        all_placeholders = {
            **prompt_parameters
        }

        # fill the prompt
        user_prompt = fill_placeholders(prompt_template, all_placeholders)
        system_prompt = fill_placeholders(SYSTEM_TEMPLATE, fields) if SYSTEM_TEMPLATE else "You are a helpful assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 4) call azure openai
        assistant_content = await call_azure_chat(messages, car.max_tokens)

        # 5) validate and possibly correct in a loop
        # attempt loop
        attempt = 1
        final_assistant_content = normalize_prices_in_text(assistant_content)
        validation_errors = validate_message(final_assistant_content, car, greeting_list, features, blacklist, car.max_tokens)

        while validation_errors and attempt < MAX_ATTEMPTS:
            # build corrective prompt that tells the model what to fix
            seller_name = getattr(car, "seller", None) if not isinstance(car, dict) else car.get("seller", None)
            preis_exists = bool(getattr(car, "preis", None) if not isinstance(car, dict) else car.get("preis", None))
            correction_prompt = build_correction_prompt(
                original_message=final_assistant_content,
                validation_errors=validation_errors,
                greeting_list=greeting_list,
                features=features,
                blacklist=blacklist,
                maxtoken=car.max_tokens,
                seller_name=seller_name,
                preis_exists=preis_exists
            )

            # prepare minimal messages for the model — system + user correction prompt
            correction_messages = [
                {"role": "system", "content": "You are a strict assistant that must follow the hard rules exactly. Return only the corrected single sentence."},
                {"role": "user", "content": correction_prompt}
            ]

            # ask the model to correct the message
            final_assistant_content = await call_azure_chat(correction_messages, car.max_tokens)

            # normalize prices in the returned text
            final_assistant_content = normalize_prices_in_text(final_assistant_content)

            # re-validate the returned sentence
            validation_errors = validate_message(final_assistant_content, car, greeting_list, features, blacklist, car.max_tokens)
            attempt += 1

        # after attempts: either valid or fail
        if validation_errors:
            # failed after retries — include final assistant content for debugging
            raise HTTPException(
                status_code=422,
                detail={
                    "validation_errors": validation_errors,
                    "assistant_content": final_assistant_content,
                    "attempts": attempt - 1
                }
            )

        # success: return validated message
        return {"message": final_assistant_content}
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No JSON file found for person_type '{car.person_type}'"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading JSON for '{car.person_type}': {e}"
        )

# @app.post("/create-json", status_code=201)
# async def create_json(payload: CreateJsonRequest):
#     try:
#         path = filepath_for(payload.name)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     try:
#         # atomic-ish write: write to temp file then rename
#         tmp = path.with_suffix(".tmp")
#         with tmp.open("w", encoding="utf-8") as f:
#             json.dump(payload.proplist, f, ensure_ascii=False, indent=2)
#             f.flush()
#         tmp.replace(path)  # overwrite existing file if exists
#     except Exception as e:
#         if path.exists():
#             try:
#                 path.unlink()
#             except Exception:
#                 pass
#         raise HTTPException(status_code=500, detail=f"Failed to write file: {e}")

#     return {
#         "message": "created/updated",
#         "filename": str(path)
#     }

@app.post("/upload-json-file")
async def upload_json_file(file: UploadFile = File(...)):
    """
    Upload a JSON file.
    1. Validates that the file is a .json file.
    2. Reads and validates the JSON content.
    3. Atomically overwrites existing file or creates a new one.
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are allowed")
    
    path = JSONS_DIR / file.filename

    try:
        # read file content
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8")
        
        # validate JSON
        content = json.loads(content_str)

        # atomic overwrite
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=JSONS_DIR
            ) as tmp:
                json.dump(content, tmp, ensure_ascii=False, indent=2)
                tmp_path = Path(tmp.name)
            
            # overwrite existing file
            os.replace(tmp_path, path)
        finally:
            # cleanup temp file if something goes wrong
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
                
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Uploaded file is not valid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload JSON file: {e}")

    return {
        "message": "JSON file uploaded successfully (created or updated)",
        "filename": str(path)
    }

@app.post("/prompts/upload-file")
async def upload_prompt_file(
    file: UploadFile = File(...),
    name: str | None = Form(None)
):
    """
    Upload a prompt file.

    1. Validates that the file is a .txt file.
    2. Reads the content.
    3. Atomically overwrites existing prompt file or creates a new one.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    prompt_name = name or Path(file.filename).stem
    path = JSONS_DIR / f"{prompt_name}.txt"

    try:
        content = (await file.read()).decode("utf-8")

        # atomic overwrite
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=JSONS_DIR
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        os.replace(tmp_path, path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload prompt: {e}")

    return {
        "message": "Prompt uploaded successfully",
        "prompt": prompt_name
    }

@app.post("/prompts/upload-system-file")
async def upload_prompt_file(file: UploadFile = File(...)):
    """
    Upload the system prompt file named 'messagetype.txt'.
    """
    # Only allow a file named 'messagetype.txt'
    if file.filename != "messagetype.txt":
        raise HTTPException(status_code=400, detail="Only 'messagetype.txt' is allowed")

    path = JSONS_DIR / "messagetype.txt"

    try:
        content = (await file.read()).decode("utf-8")

        # atomic overwrite
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=JSONS_DIR
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        os.replace(tmp_path, path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload prompt: {e}")

    return {
        "message": "Prompt uploaded successfully",
        "prompt": "messagetype"
    }



@app.get("/list-jsons")
async def list_jsons():
    """
    List all JSON files in the JSONS_DIR.

    """
    files = [p.name for p in JSONS_DIR.glob("*.json")]
    return {"files": files}


@app.get("/prompts")
async def list_prompts():
    """
    List all prompt text files in the JSONS_DIR.

    """
    if not JSONS_DIR.exists():
        return {"prompts": []}

    prompts = [
        p.stem
        for p in JSONS_DIR.iterdir()
        if p.is_file() and p.suffix == ".txt"
    ]

    return {
        "count": len(prompts),
        "prompts": sorted(prompts)
    }


@app.get("/read-json/{name}")
async def read_json(name: str):
    """
    Read a JSON file by name (without .json extension)
    """
    try:
        data = load_json(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"name": name, "proplist": data}

@app.get("/prompts/{name}")
async def view_prompt(name: str):
    """
    Get the content of a prompt by name (without .txt extension)
    """
    file_path = JSONS_DIR / f"{name}.txt"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")
    
    content = file_path.read_text(encoding="utf-8")
    return {"name": name, "content": content}

@app.get("/download-json/{filename}")
def download_json(filename: str):
    """
    Download a JSON file by filename.
    """
     # Ensure filename ends with .json
    if not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = JSONS_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        media_type="application/json",
        filename=filename
    )


@app.get("/prompts/download/{prompt_name}")
def download_prompt(prompt_name: str):
    """
    Download a prompt text file by name.
    """
    if not prompt_name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = JSONS_DIR / prompt_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Prompt not found")

    return FileResponse(
        path=file_path,
        media_type="text/plain",
        filename=prompt_name
    )

@app.delete("/delete-json/{name}", status_code=200)
async def delete_json(name: str = FastAPIPath(..., description="Name of the JSON file to delete")):
    """
    Delete a JSON file by name.
    """
    try:

        safe_name = normalize_filename(name)
        path = filepath_for(safe_name)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File '{path.name}' not found.")

    try:
        path.unlink()  # delete the file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

    return {"message": "deleted", "filename": str(path)}


@app.delete("/prompts/delete/{filename}")
async def delete_prompt(filename: str):
    """""
    Delete a prompt file by name.
    """
    # Ensure filename ends with .txt
    if not filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files can be deleted.")
    
    file_path = os.path.join(JSONS_DIR, filename)
    
    # Check if file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    # Delete the file
    try:
        os.remove(file_path)
        return {"message": f"{filename} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")