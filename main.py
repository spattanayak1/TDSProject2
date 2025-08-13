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
from zipfile import ZipFile
import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image
import docx
import matplotlib.pyplot as plt
import networkx as nx

app = FastAPI()

# ==== CONFIG ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a senior data analyst. You will receive messages from a manager that may contain tasks, questions, objectives, and sometimes data sources (e.g., CSV, URLs, or descriptions).

Your job:
1. Extract all tasks and objectives.
2. Analyze and interpret the request with precision.
3. Break down into structured components.
4. Return ONLY valid JSON with the exact schema requested or a logical minimal JSON if none given.
5. For plots, return the data in a way that allows the backend to generate them (include chart_type, x_data, y_data, labels if needed).
6. Do NOT return explanations outside of JSON.
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
def encode_image_to_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    image_bytes = buf.getvalue()

    while len(image_bytes) > 100_000:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        image_bytes = buf.getvalue()

    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

# ==== GENERIC PLOT GENERATOR ====
def generate_chart(chart_type, x_data=None, y_data=None, labels=None, **kwargs):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if chart_type == "scatter":
        ax.scatter(x_data, y_data, **kwargs)
    elif chart_type == "line":
        ax.plot(x_data, y_data, **kwargs)
    elif chart_type == "bar":
        ax.bar(x_data, y_data, **kwargs)
    elif chart_type == "hist":
        ax.hist(x_data, bins=kwargs.get("bins", 10), **kwargs)
    elif chart_type == "pie":
        ax.pie(y_data, labels=labels, autopct='%1.1f%%', **kwargs)
    else:
        plt.close(fig)
        return None

    ax.set_xlabel(kwargs.get("xlabel", ""))
    ax.set_ylabel(kwargs.get("ylabel", ""))
    ax.set_title(kwargs.get("title", chart_type.capitalize()))

    fig.tight_layout()
    return encode_image_to_data_uri(fig)

# ==== GPT HELPERS ====
def generate_code(prompt: str) -> str:
    full_prompt = (
        "You are a Python automation expert.\n"
        "Write a complete Python script to solve the task:\n"
        f"{prompt}\n\n"
        "Requirements:\n"
        "- Final result in variable result_json.\n"
        "- Match exactly the requested JSON format.\n"
        "- No prints or logs.\n"
        "- Only use built-in Python libraries.\n"
    )
    response = client.responses.create(model="gpt-5-nano", input=full_prompt)
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
    for _ in range(max_attempts):
        code = generate_code(prompt if not error_message else f"Fix error:\n{error_message}\nTask:\n{prompt}")
        json_result, error, executed_code = execute_code(code)
        if not error:
            return json_result, executed_code, None
        error_message = error
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

        json_result, executed_code, error_message = await feedback_loop(prompt)

        if json_result:
            # Handle dynamic chart generation
            if "plots" in json_result and isinstance(json_result["plots"], list):
                for plot in json_result["plots"]:
                    chart_uri = generate_chart(
                        chart_type=plot.get("type"),
                        x_data=plot.get("x"),
                        y_data=plot.get("y"),
                        labels=plot.get("labels"),
                        **plot.get("kwargs", {})
                    )
                    plot["image"] = chart_uri
            return JSONResponse(content=json_result)

        fallback_response = call_openai_chat(prompt)
        try:
            parsed = json.loads(fallback_response)
            if "plots" in parsed and isinstance(parsed["plots"], list):
                for plot in parsed["plots"]:
                    chart_uri = generate_chart(
                        chart_type=plot.get("type"),
                        x_data=plot.get("x"),
                        y_data=plot.get("y"),
                        labels=plot.get("labels"),
                        **plot.get("kwargs", {})
                    )
                    plot["image"] = chart_uri
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(status_code=500, content={"error": "Invalid JSON from fallback", "response": fallback_response})

    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb_str})
