import os
import io
import json
import base64
import traceback
from zipfile import ZipFile
from typing import Dict

import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
import networkx as nx
import pdfplumber
import docx
from PIL import Image
import xml.etree.ElementTree as ET

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

# ==== CONFIG ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a senior data analyst. You receive tasks, numeric questions, and sometimes files.
Return only JSON with numeric results, arrays, or plots. Plots must have 'plot_type' and 'data' + optional 'options'.
Do not include any commentary or markdown.
"""

# ==== FILE PROCESSING ====
def process_file(fname: str, content: bytes):
    try:
        lower_name = fname.lower()
        if lower_name.endswith((".png", ".jpg", ".jpeg")):
            return pytesseract.image_to_string(Image.open(io.BytesIO(content))).strip()
        elif lower_name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content)).to_dict(orient="records")
        elif lower_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content)).to_dict(orient="records")
        elif lower_name.endswith(".json"):
            return json.loads(content.decode("utf-8", errors="ignore"))
        elif lower_name.endswith(".xml"):
            tree = ET.ElementTree(ET.fromstring(content.decode("utf-8", errors="ignore")))
            return ET.tostring(tree.getroot(), encoding="unicode")
        elif lower_name.endswith(".pdf"):
            text = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text.append(t)
            return "\n".join(text).strip()
        elif lower_name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs])
        elif lower_name.endswith(".zip"):
            with ZipFile(io.BytesIO(content)) as zf:
                return {"zip_contents": zf.namelist()}
        elif lower_name.endswith(".txt"):
            return content.decode("utf-8", errors="ignore")
        else:
            return content.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"<Error processing {fname}: {str(e)}>"

# ==== IMAGE ENCODING ====
def encode_fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

# ==== DYNAMIC CHART RENDERING ====
def render_dynamic_chart(data, options=None):
    if options is None:
        options = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    kind = options.get("kind", "line")
    
    try:
        if kind == "line":
            ax.plot(data.get("x", []), data.get("y", []), color=options.get("color", "red"), linewidth=2)
        elif kind == "scatter":
            ax.scatter(data.get("x", []), data.get("y", []), color=options.get("color", "blue"))
        elif kind == "bar":
            ax.bar(data.get("x", []), data.get("y", []), color=options.get("color", "green"))
        elif kind == "histogram":
            ax.hist(data.get("values", []), bins=options.get("bins", 10), color=options.get("color", "orange"), edgecolor="black")
        elif kind == "pie":
            ax.pie(data.get("values", []), labels=data.get("labels", []), autopct="%1.1f%%")
        elif kind == "network":
            edges = data
            G = nx.Graph()
            G.add_edges_from(edges)
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=800, edge_color="gray", ax=ax)
        elif kind == "degree_histogram":
            edges = data
            G = nx.Graph()
            G.add_edges_from(edges)
            degrees = [deg for _, deg in G.degree()]
            ax.bar(range(len(degrees)), degrees, color="green")
            ax.set_xlabel("Node Index")
            ax.set_ylabel("Degree")
            ax.set_title("Degree Distribution")
        elif kind == "regression":
            import numpy as np
            x = np.array(data.get("x", []))
            y = np.array(data.get("y", []))
            ax.scatter(x, y, color="blue", label="Data points")
            if len(x) > 1:
                coef = np.polyfit(x, y, 1)
                fit_line = np.poly1d(coef)
                ax.plot(x, fit_line(x), color="red", linewidth=2, label="Fit line")
            ax.set_xlabel(options.get("xlabel", "X"))
            ax.set_ylabel(options.get("ylabel", "Y"))
            ax.set_title(options.get("title", "Regression Chart"))
            ax.legend()
        else:
            ax.plot(data.get("x", []), data.get("y", []))
            ax.set_title(options.get("title", kind))
    except Exception as e:
        plt.close(fig)
        return f"<Error rendering plot: {str(e)}>"
    
    ax.set_xlabel(options.get("xlabel", ""))
    ax.set_ylabel(options.get("ylabel", ""))
    ax.set_title(options.get("title", ""))
    fig.tight_layout()
    return encode_fig_to_base64(fig)

# ==== EXECUTE GENERATED PYTHON CODE ====
def execute_code(code: str):
    try:
        namespace = {}
        exec(code, namespace)
        if "result_json" in namespace:
            return namespace["result_json"], None
        return None, "No 'result_json' variable found"
    except Exception as e:
        return None, f"Execution error: {str(e)}"

async def feedback_loop(prompt: str, max_attempts=4):
    error_message = ""
    for attempt in range(max_attempts):
        full_prompt = f"Fix the code based on this error:\n{error_message}\nTask:\n{prompt}" if error_message else prompt
        response = client.responses.create(
            model="gpt-5-nano",
            input=f"Write Python code for this task. Final result must be in result_json:\n{full_prompt}"
        )
        code = response.output_text.strip()
        result, error = execute_code(code)
        if error is None:
            return result, code, None
        error_message = f"{error}\n\nCode:\n{code}"
    return None, None, error_message

# ==== OPENAI FALLBACK ====
def call_openai_chat(prompt: str):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        
    )
    return response.choices[0].message.content.strip()

# ==== API ENDPOINT ====
@app.post("/api/")
async def analyze(request: Request):
    try:
        form = await request.form()
        if "questions.txt" not in form:
            return JSONResponse(status_code=400, content={"error": "Missing questions.txt file"})

        questions_content = await form["questions.txt"].read()
        prompt = questions_content.decode("utf-8", errors="ignore").strip()

        files_data = {}
        for key, value in form.items():
            if key != "questions.txt" and hasattr(value, "filename"):
                files_data[value.filename] = process_file(value.filename, await value.read())

        if files_data:
            prompt += "\n\nAttached Files:\n"
            for fname, data in files_data.items():
                prompt += f"- {fname}:\n{data}\n"

        # Try feedback loop first (Python code execution)
        json_result, executed_code, error_message = await feedback_loop(prompt)
        if json_result:
            for k, v in list(json_result.items()):
                if isinstance(v, dict) and v.get("plot_type"):
                    plot_type = v.pop("plot_type")
                    data = v.pop("data", v)
                    options = v.pop("options", {})
                    json_result[k] = render_dynamic_chart(data, options)
            return JSONResponse(content=json_result)

        # Fallback to direct OpenAI JSON response
        fallback_response = call_openai_chat(prompt)
        try:
            parsed = json.loads(fallback_response)
            for k, v in list(parsed.items()):
                if isinstance(v, dict) and v.get("plot_type"):
                    plot_type = v.pop("plot_type")
                    data = v.pop("data", v)
                    options = v.pop("options", {})
                    parsed[k] = render_dynamic_chart(data, options)
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(status_code=500, content={"error": "OpenAI response not valid JSON", "response": fallback_response})

    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb_str})
