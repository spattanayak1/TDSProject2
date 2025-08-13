import os
import traceback
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
import json
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

SYSTEM_PROMPT = """..."""  # unchanged from your version

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

# ==== GRAPH PLOTTING ====
def generate_network_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

def generate_degree_histogram(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    degrees = [deg for _, deg in G.degree()]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(degrees)), degrees, color="green")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Degree")
    ax.set_title("Degree Distribution")
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

def generate_line_chart(dates, values, color="red", title="Line Chart"):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values})
    df = df.sort_values("date")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["date"], df["value"], color=color, linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

def generate_histogram(values, color="orange", title="Histogram"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=10, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

# ==== GPT HELPERS ====
def generate_code(prompt: str) -> str:
    full_prompt = (
        "You are a Python automation expert. "
        "Write a complete Python script to do the following task:\n"
        f"{prompt}\n\n"
        "- Assign the final result to a variable named result_json.\n"
        "- Output ONLY the Python code."
    )
    response = client.responses.create(model="gpt-5-nano", input=full_prompt)
    return response.output_text.strip()

def execute_code(code: str):
    try:
        namespace = {}
        exec(code, namespace)
        if "result_json" in namespace:
            return namespace["result_json"], None, code
        return None, "No 'result_json' variable found", code
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

# ==== CHART AUTO-DETECT ====
def auto_generate_charts(json_result: dict):
    for key, value in list(json_result.items()):
        if isinstance(value, list) and value and isinstance(value[0], dict):
            keys_lower = [k.lower() for k in value[0].keys()]
            if "date" in keys_lower and any(isinstance(v, (int, float)) for v in value[0].values()):
                numeric_col = [k for k in value[0].keys() if k.lower() != "date"][0]
                json_result[f"{key}_chart"] = generate_line_chart(
                    [row["date"] for row in value],
                    [row[numeric_col] for row in value],
                    color="red" if "temperature" in key.lower() else "blue",
                    title=f"{key} over Time"
                )
                del json_result[key]
        elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
            json_result[f"{key}_histogram"] = generate_histogram(
                value,
                color="orange" if "precip" in key.lower() else "gray",
                title=f"{key} Histogram"
            )
            del json_result[key]
    return json_result

# ==== API ENDPOINT ====
@app.post("/api/")
async def analyze(request: Request):
    try:
        form = await request.form()
        if "questions.txt" not in form:
            return JSONResponse(status_code=400, content={"error": "Missing questions.txt file"})

        prompt = (await form["questions.txt"].read()).decode("utf-8", errors="ignore").strip()
        files_data = {}
        for key, value in form.items():
            if key != "questions.txt" and hasattr(value, "filename"):
                files_data[value.filename] = process_file(value.filename, await value.read())

        if files_data:
            prompt += "\n\nAttached Files:\n" + "\n".join(f"- {fname}: {data}" for fname, data in files_data.items())

        json_result, executed_code, error_message = await feedback_loop(prompt, max_attempts=4)
        if json_result is not None:
            if "edges" in json_result:
                json_result["network_graph"] = generate_network_graph(json_result["edges"])
                json_result["degree_histogram"] = generate_degree_histogram(json_result["edges"])
                del json_result["edges"]

            json_result = auto_generate_charts(json_result)
            return JSONResponse(content=json_result)

        fallback_response = call_openai_chat(prompt)
        try:
            parsed = json.loads(fallback_response)
            parsed = auto_generate_charts(parsed)
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(status_code=500, content={"error": "Invalid JSON", "response": fallback_response})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
