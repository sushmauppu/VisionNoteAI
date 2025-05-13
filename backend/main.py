from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
import easyocr
import google.generativeai as genai
from PIL import Image
from fpdf import FPDF
from io import BytesIO
import numpy as np
import os

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# Initialize Gemini Model
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize FastAPI
app = FastAPI(title="Smart Notes API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR Reader Initialization
ocr_reader = easyocr.Reader(['en'])

@app.get("/")
async def root():
    return {"message": "Welcome to Smart Notes API"}


def extract_text(image_bytes: bytes) -> str:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        results = ocr_reader.readtext(np.array(image))
        extracted = ' '.join([text[1] for text in results]).strip()
        if not extracted:
            raise ValueError("No text could be extracted from the image.")
        return extracted
    except Exception as e:
        raise RuntimeError(f"OCR extraction failed: {str(e)}")


def generate_study_notes(topic: str, extracted_text: str) -> str:
    try:
        prompt = f"""
        You are a helpful academic assistant. A student is studying the topic: "{topic}".
        Extracted notes: "{extracted_text}".
        Generate clean, concise, and well-organized study notes with:
        - Clear headings and subheadings.
        - Logical structure and easy-to-read format.
        - No comments, debugging text, or extraneous symbols.
        """
        response = model.generate_content(prompt)
        notes = response.text.replace('*', '').replace('•', '').strip()

        # Improve formatting
        for heading in ['#', '##', '###']:
            notes = notes.replace(heading, '\n\n')
        notes = '\n\n'.join(line.strip() for line in notes.splitlines() if line.strip())
        return notes
    except Exception as e:
        raise RuntimeError(f"Gemini API generation failed: {str(e)}")


def generate_pdf(notes_text: str) -> BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    margin = 15
    pdf.set_left_margin(margin)
    pdf.set_right_margin(margin)
    pdf.ln(5)

    # Split notes into sections by blank lines
    sections = [sec.strip() for sec in notes_text.split('\n\n') if sec.strip()]
    for sec in sections:
        words = sec.split()
        # Set font based on heading depth
        if len(words) <= 3:
            pdf.set_font('Times', 'B', 16)  # Main heading
        elif len(words) <= 6:
            pdf.set_font('Times', 'B', 14)  # Subheading
        else:
            pdf.set_font('Times', '', 12)   # Body text
        pdf.multi_cell(0, 8, sec)
        pdf.ln(4)

    output = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    output.write(pdf_bytes)
    output.seek(0)
    return output


@app.post("/generate-notes")
async def generate_notes(file: UploadFile = File(...), topic: str = Form(...)):
    try:
        # Validate file size and type
        file_content = await file.read()
        if len(file_content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")

        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only JPEG, JPG, and PNG are allowed.")

        # Process image and generate notes
        extracted_text = extract_text(file_content)
        notes = generate_study_notes(topic, extracted_text)

        return {"notes": notes}

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate-notes-pdf")
async def generate_notes_pdf(file: UploadFile = File(...), topic: str = Form(...)):
    try:
        content = await file.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        extracted = extract_text(content)
        notes = generate_study_notes(topic, extracted)
        pdf_file = generate_pdf(notes)
        return StreamingResponse(
            pdf_file,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={topic.replace(' ', '_')}.pdf"}
        )

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
