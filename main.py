import os
import traceback
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
import json
import csv
import io
import base64
import xml.etree.ElementTree as ET
import PyPDF2
from zipfile import ZipFile
import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image
import docx
import matplotlib.pyplot as plt

app = FastAPI()

# ==== CONFIG ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a senior data analyst. You will receive messages from a manager that may contain a mix of tasks, business questions, objectives, and sometimes a data source or description of data (e.g., a CSV file, database schema, table summary, or a URL to scrape data from).

Your job is to:
1. Carefully read and extract all requested tasks and objectives.
2. Analyze and interpret the request with precision.
3. Break down the tasks into specific, structured components.
4. Format your entire response strictly in JSON following the exact response schema given in the message (if provided).
5. If no response schema is provided, infer a logical and minimal JSON structure that matches the questions asked.

Instructions:
- Do not include any explanation, commentary, markdown formatting, or natural language text outside of the JSON.
- Return only a valid, well-formatted JSON object (or array if specified).
- Do not make assumptions beyond what is asked in the request.
- If a data source is mentioned, ensure it is included in the task context.
- Every task must include a clear objective, any dependencies (like datasets or URLs), and expected output format if described.
- You may be asked to do data scraping, correlation, visualizations, or numerical analysis.

Response formatting rules:
- Always return the JSON exactly as instructed by the request.
- If a response structure is provided (e.g., JSON array or object), strictly follow it.
- If a plot is requested, return a base64-encoded data URI string (e.g., "data:image/png;base64,..."), and ensure it's under 100,000 bytes when instructed.

Only return valid JSON. Be accurate, structured, and concise.
Don't give sentences, give me just the data, numbers and the things User asked for.
"""

# ==== FILE HANDLING ====
def process_file(fname: str, content: bytes):
    lower_name = fname.lower()
    try:
        if lower_name.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)
            return text.strip()
        elif lower_name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            return df.to_dict(orient="records")
        elif lower_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
            return df.to_dict(orient="records")
        elif lower_name.endswith(".json"):
            return json.loads(content.decode("utf-8", errors="ignore"))
        elif lower_name.endswith(".xml"):
            xml_text = content.decode("utf-8", errors="ignore")
            tree = ET.ElementTree(ET.fromstring(xml_text))
            return ET.tostring(tree.getroot(), encoding="unicode")
        elif lower_name.endswith(".pdf"):
            pdf_text = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text.append(page_text)
            return "\n".join(pdf_text).strip()
        elif lower_name.endswith(".docx"):
            doc_obj = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc_obj.paragraphs])
        elif lower_name.endswith(".zip"):
            try:
                with ZipFile(io.BytesIO(content)) as zf:
                    return {"zip_contents": zf.namelist()}
            except:
                return "<ZIP file could not be processed>"
        elif lower_name.endswith(".txt"):
            return content.decode("utf-8", errors="ignore")
        else:
            try:
                return content.decode("utf-8", errors="ignore")
            except:
                return f"<Binary file {fname}, {len(content)} bytes>"
    except Exception as e:
        return f"<Error processing {fname}: {str(e)}>"

# ==== IMAGE ENCODING ====
def encode_image_to_data_uri(image_obj):
    """Convert a PIL Image, matplotlib figure, or image bytes to base64 data URI under 100KB."""
    # Handle matplotlib figure
    if "matplotlib" in str(type(image_obj)):
        buf = io.BytesIO()
        image_obj.savefig(buf, format="PNG", bbox_inches="tight")
        plt.close(image_obj)
        image_obj = Image.open(io.BytesIO(buf.getvalue()))

    if isinstance(image_obj, Image.Image):
        buffer = io.BytesIO()
        image_obj.save(buffer, format="PNG", optimize=True)
        image_bytes = buffer.getvalue()
    elif isinstance(image_obj, bytes):
        image_bytes = image_obj
        image_obj = Image.open(io.BytesIO(image_bytes))
    else:
        raise ValueError("Unsupported image format for base64 encoding")

    quality = 95
    while len(image_bytes) > 100_000 and quality > 10:
        buffer = io.BytesIO()
        image_obj.save(buffer, format="PNG", optimize=True, quality=quality)
        image_bytes = buffer.getvalue()
        quality -= 5

    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

# ==== GPT HELPERS ====
def generate_code(prompt: str) -> str:
    full_prompt = (
        "You are a Python automation expert. "
        "Write a complete Python script to do the following task:\n"
        f"{prompt}\n\n"
        "The script must:\n"
        "- Assign the final result to a variable named result_json.\n"
        "- The format of result_json must match exactly the format requested in the question.\n"
        "- Do not print anything. No explanations, no markdown, no logs.\n"
        "- Use only built-in Python libraries (no pip installs).\n"
        "- Output ONLY the Python code, nothing else."
    )

    response = client.responses.create(
        model="gpt-5-nano",
        input=full_prompt
    )
    return response.output_text.strip()

def execute_code(code: str):
    try:
        namespace = {}
        exec(code, namespace)
        if "result_json" in namespace:
            return namespace["result_json"], None, code
        if "main" in namespace and callable(namespace["main"]):
            result = namespace["main"]()
            if result is None and "result_json" in namespace:
                return namespace["result_json"], None, code
            return result, None, code
        return None, "No 'result_json' variable or callable main() found", code
    except Exception as e:
        return None, f"Execution error: {str(e)}", code

async def feedback_loop(prompt: str, max_attempts=4):
    error_message = ""
    for attempt in range(max_attempts):
        full_prompt = f"Fix the code based on this error:\n{error_message}\nTask:\n{prompt}" if error_message else prompt
        code = generate_code(full_prompt)
        json_result, error, executed_code = execute_code(code)
        if error is None:
            return json_result, executed_code, None
        error_message = f"{error}\n\nCode:\n{code}"
    return None, None, error_message

def call_openai_chat(prompt: str):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ==== API ENDPOINT ====
@app.post("/api/")
async def analyze(request: Request):
    try:
        form = await request.form()

        if "questions.txt" not in form:
            return JSONResponse(status_code=400, content={"error": "Missing questions.txt file"})

        questions_file = form["questions.txt"]
        questions_content = await questions_file.read()
        prompt = questions_content.decode("utf-8", errors="ignore").strip()

        files_data = {}
        for key, value in form.items():
            if key != "questions.txt" and hasattr(value, "filename"):
                file_bytes = await value.read()
                files_data[value.filename] = process_file(value.filename, file_bytes)

        if files_data:
            prompt += "\n\nAttached Files:\n"
            for fname, data in files_data.items():
                prompt += f"- {fname}:\n{data}\n"

        json_result, executed_code, error_message = await feedback_loop(prompt, max_attempts=4)

        if json_result is not None:
            # Flatten so keys are at top level, and encode images
            if isinstance(json_result, dict):
                flat_result = {}
                for k, v in json_result.items():
                    if isinstance(v, (Image.Image, bytes)) or "matplotlib" in str(type(v)):
                        flat_result[k] = encode_image_to_data_uri(v)
                    else:
                        flat_result[k] = v
                return JSONResponse(content=flat_result)
            elif isinstance(json_result, list):
                flat_list = []
                for item in json_result:
                    if isinstance(item, (Image.Image, bytes)) or "matplotlib" in str(type(item)):
                        flat_list.append(encode_image_to_data_uri(item))
                    else:
                        flat_list.append(item)
                return JSONResponse(content={"result": flat_list})

        # Fallback to chat
        fallback_response = call_openai_chat(prompt)
        try:
            parsed = json.loads(fallback_response)
            # Flatten fallback too
            if isinstance(parsed, dict):
                return JSONResponse(content=parsed)
            return JSONResponse(content={"result": parsed})
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=500,
                content={"error": "Fallback OpenAI response is not valid JSON", "response": fallback_response}
            )

    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb_str})
