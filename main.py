import os
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from openai import OpenAI
import json

app = FastAPI()

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
            return json_result, executed_code, None  # Success, no error
        error_message = f"{error}\n\nCode:\n{code}"
    return None, None, error_message  # Failure after max attempts

def call_openai_chat(prompt: str):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    assistant_reply = response.choices[0].message.content.strip()
    return assistant_reply


@app.post("/api/")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        prompt = contents.decode("utf-8").strip()

        # Try to generate and execute Python code up to 4 times
        json_result, executed_code, error_message = await feedback_loop(prompt, max_attempts=4)

        if json_result is not None:
            # Success, return JSON result
            return JSONResponse(content={"data": json_result})

        # After 4 failed attempts, call OpenAI chat completion fallback (senior data analyst style)
        fallback_response = call_openai_chat(prompt)

        # Try to parse fallback response as JSON and return
        try:
            parsed = json.loads(fallback_response)
            return JSONResponse(content={"data": parsed})
        except json.JSONDecodeError:
            # If fallback output is not valid JSON, return raw string with error status
            return JSONResponse(
                status_code=500,
                content={"error": "Fallback OpenAI response is not valid JSON", "response": fallback_response}
            )

    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb_str})
