import os
import traceback
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Request
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
import csv

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
def encode_image_to_data_uri(fig):
    # Initial save
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    image_bytes = buf.getvalue()

    # Reduce size if needed
    dpi = 120
    while len(image_bytes) > 100_000 and dpi > 20:
        buf = io.BytesIO()
        dpi -= 10
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        image_bytes = buf.getvalue()

    plt.close(fig)  # Only close after we're done
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


# ==== SPECIFIC CHART HELPERS ====
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

def generate_cumulative_sales_chart(dates, sales):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "sales": sales})
    df = df.sort_values("date")
    df["cumulative_sales"] = df["sales"].cumsum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["date"], df["cumulative_sales"], color="red", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Sales")
    ax.set_title("Cumulative Sales Over Time")
    ax.grid(True)
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

def generate_precip_histogram(precip_values):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(precip_values, bins=10, color="orange", edgecolor="black")
    ax.set_xlabel("Precipitation Amount")
    ax.set_ylabel("Frequency")
    ax.set_title("Precipitation Histogram")
    ax.grid(True)
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

def generate_line_chart(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [row["x"] for row in data]
    y = [row["y"] for row in data]
    ax.plot(x, y, marker="o", color="blue")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Line Chart")
    ax.grid(True)
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

def generate_pie_chart(data):
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = [row["label"] for row in data]
    values = [row["value"] for row in data]
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Pie Chart")
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

def generate_regression_plot(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data["x"], data["y"], color="blue", label="Actual")
    ax.plot(data["x"], data["y_pred"], color="red", linewidth=2, label="Predicted")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Regression Plot")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

# ==== GENERIC CHART HANDLER ====
def generate_generic_chart(chart_info):
    fig, ax = plt.subplots(figsize=(6, 4))
    chart_type = chart_info.get("type", "").lower()
    data = chart_info.get("data", [])

    if chart_type == "line":
        x = [row["x"] for row in data]
        y = [row["y"] for row in data]
        ax.plot(x, y, marker="o", color="blue")
    elif chart_type == "bar":
        x = [row["x"] for row in data]
        y = [row["y"] for row in data]
        ax.bar(x, y, color="skyblue", edgecolor="black")
    elif chart_type == "scatter":
        x = [row["x"] for row in data]
        y = [row["y"] for row in data]
        ax.scatter(x, y, color="purple")
    elif chart_type == "pie":
        labels = [row["label"] for row in data]
        values = [row["value"] for row in data]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    else:
        ax.text(0.5, 0.5, f"Chart type '{chart_type}' not supported",
                ha='center', va='center', fontsize=12, color="red")
        ax.set_axis_off()

    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return encode_image_to_data_uri(fig)

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
        ]
        
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
            # Existing chart handlers
            if "edges" in json_result and isinstance(json_result["edges"], list):
                json_result["network_graph"] = generate_network_graph(json_result["edges"])
                json_result["degree_histogram"] = generate_degree_histogram(json_result["edges"])
                del json_result["edges"]

            if "cumulative_sales_data" in json_result:
                dates = [row["date"] for row in json_result["cumulative_sales_data"]]
                sales = [row["sales"] for row in json_result["cumulative_sales_data"]]
                json_result["cumulative_sales_chart"] = generate_cumulative_sales_chart(dates, sales)
                del json_result["cumulative_sales_data"]

            if "precip_data" in json_result:
                json_result["precip_histogram"] = generate_precip_histogram(json_result["precip_data"])
                del json_result["precip_data"]

            # New fixed handlers
            if "line_chart_data" in json_result:
                json_result["line_chart"] = generate_line_chart(json_result["line_chart_data"])
                del json_result["line_chart_data"]

            if "pie_chart_data" in json_result:
                json_result["pie_chart"] = generate_pie_chart(json_result["pie_chart_data"])
                del json_result["pie_chart_data"]

            if "regression_data" in json_result:
                json_result["regression_plot"] = generate_regression_plot(json_result["regression_data"])
                del json_result["regression_data"]

            # Generic handler for future chart types
            if "chart" in json_result and isinstance(json_result["chart"], dict):
                json_result["chart_image"] = generate_generic_chart(json_result["chart"])
                del json_result["chart"]

            return JSONResponse(content=json_result)

        fallback_response = call_openai_chat(prompt)
        try:
            parsed = json.loads(fallback_response)
            if "edges" in parsed and isinstance(parsed["edges"], list):
                parsed["network_graph"] = generate_network_graph(parsed["edges"])
                parsed["degree_histogram"] = generate_degree_histogram(parsed["edges"])
                del parsed["edges"]
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(status_code=500, content={"error": "Fallback OpenAI response is not valid JSON", "response": fallback_response})

    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb_str})
