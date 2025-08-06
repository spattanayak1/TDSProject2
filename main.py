import os
import sys
import re
import traceback
import subprocess
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

# Load environment variables
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


model = genai.GenerativeModel("gemini-1.5-flash")
app = FastAPI()

def generate_code_and_dependencies(prompt):
    full_prompt = (
        f"Write a complete Python script to do the following:\n{prompt}\n\n"
        "Return two outputs:\n"
        "1. A Python list of all external packages needed (only the package names).\n"
        "2. The full code (no markdown formatting).\n\n"
        "Format your output exactly like:\n"
        "PACKAGES:\n['package1', 'package2']\n\nCODE:\n<code starts here>"
    )
    response = model.generate_content(full_prompt)
    return response.text

def parse_packages_and_code(response_text):
    try:
        packages_section = re.search(r'PACKAGES:\n(\[.*?\])', response_text, re.DOTALL).group(1)
        code_section = re.search(r'CODE:\n(.+)', response_text, re.DOTALL).group(1)
        packages = eval(packages_section.strip())
        return packages, code_section.strip()
    except Exception as e:
        raise ValueError(f"Parsing error: {e}\n\nResponse was:\n{response_text}")

def install_packages(package_list):
    for pkg in package_list:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError:
            print(f"Failed to install: {pkg}")

def execute_code(code):
    local_vars = {}
    try:
        exec(code, local_vars)

        # Try to find any DataFrame in the result
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, pd.DataFrame):
                print(f"✅ Found DataFrame in variable: {var_name}")
                return var_value, None, code

        return None, "Code executed, but no DataFrame found in variables.", code
    except Exception:
        return None, traceback.format_exc(), code



def feedback_loop(prompt, max_attempts=3):
    error_message = ""
    for attempt in range(max_attempts):
        full_prompt = f"Previous error:\n{error_message}\n\n{prompt}" if error_message else prompt
        response_text = generate_code_and_dependencies(full_prompt)

        try:
            packages, code = parse_packages_and_code(response_text)
        except Exception as e:
            error_message = f"Error parsing packages/code: {str(e)}"
            continue

        df, error = execute_code(code)
        if df is not None:
            return df

        error_message = f"Error during code execution: {error}"

    # After all attempts fail, raise the last known error
    raise RuntimeError(f"Failed after multiple attempts.\nLast error: {error_message}")


@app.post("/api/")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        prompt = contents.decode("utf-8").strip()
        df = feedback_loop(prompt)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
            df.to_json(tmp.name, orient="records", indent=2)
            tmp.flush()
            tmp.seek(0)
            json_output = pd.read_json(tmp.name)

        return JSONResponse(content=json_output.to_dict(orient="records"))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

