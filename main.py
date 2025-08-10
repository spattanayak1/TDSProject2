import os
import traceback
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from openai import OpenAI
import json
import csv
import io
import base64
import xml.etree.ElementTree as ET
import PyPDF2  # Built-in to some Python environments, if not, handle as bytes
from zipfile import ZipFile

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
    """Detect file type and return structured content for prompt."""
    lower_name = fname.lower()
    try:
        if lower_name.endswith(".png") or lower_name.endswith(".jpg") or lower_name.endswith(".jpeg"):
            return f"data:image/{'png' if 'png' in lower_name else 'jpeg'};base64,{base64.b64encode(content).decode('utf-8')}"
        elif lower_name.endswith(".csv"):
            csv_text = content.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(csv_text))
            return list(reader)
        elif lower_name.endswith(".json"):
            return json.loads(content.decode("utf-8", errors="ignore"))
        elif lower_name.endswith(".xml"):
            xml_text = content.decode("utf-8", errors="ignore")
            tree = ET.ElementTree(ET.fromstring(xml_text))
            return ET.tostring(tree.getroot(), encoding="unicode")
        elif lower_name.endswith(".pdf"):
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
                return text.strip()
            except:
                return "<PDF could not be extracted>"
        elif lower_name.endswith(".zip"):
            try:
                with ZipFile(io.BytesIO(content)) as zf:
                    file_list = zf.namelist()
                    return {"zip_contents": file_list}
            except:
                return "<ZIP file could not be processed>"
        elif lower_name.endswith(".txt"):
            return content.decode("utf-8", errors="ignore")
        else:
            # Generic fallback
            try:
                return content.decode("utf-8", errors="ignore")
            except:
                return f"<Binary file {fname}, {len(content)} bytes>"
    except Exception as e:
        return f"<Error processing {fname}: {str(e)}>"

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
async def analyze(
    questions: UploadFile = File(..., description="The main questions.txt file"),
    extra_files: Optional[List[UploadFile]] = File(None, description="Optional additional files (any type)")
):
    try:
        # Read main questions.txt
        questions_content = await questions.read()
        prompt = questions_content.decode("utf-8", errors="ignore").strip()

        # Process extra files
        files_data = {}
        if extra_files:
            for file in extra_files:
                content = await file.read()
                files_data[file.filename] = process_file(file.filename, content)

        # Append processed file info & data to prompt
        if files_data:
            prompt += "\n\nAttached Files:\n"
            for fname, data in files_data.items():
                prompt += f"- {fname}:\n{data}\n"

        # Try code-gen + execution loop
        json_result, executed_code, error_message = await feedback_loop(prompt, max_attempts=4)

        if json_result is not None:
            return JSONResponse(content={"data": json_result})

        # Fallback to chat mode
        fallback_response = call_openai_chat(prompt)
        try:
            parsed = json.loads(fallback_response)
            return JSONResponse(content={"data": parsed})
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=500,
                content={"error": "Fallback OpenAI response is not valid JSON", "response": fallback_response}
            )

    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb_str})
