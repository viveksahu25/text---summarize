from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch 
import re
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# initalize our fastapi app 

app = FastAPI(title="Text Summarizer App", description="text summarization usin g T5", version="1.0")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
#  model & tokenizer
 
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

#  device 

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# print("device", device)
model.to(device)
model.eval()

#  templating 
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

#  input schema for dialogue => string

class DialogueInput(BaseModel):
    dialogue: str

# import re

def clean_data(text):
  text = re.sub(r"\r\n", " ", text)  #line ko replace kar rhi ha
  text = re.sub(r"\s+"," ", text)   # space ko remove kr rhi ha
  text = re.sub(r"<.*?>"," ", text)   # remove html tag
  text = text.strip().lower()
  return text



def summarize_dialogue(dialogue : str) -> str:
  dialogue = clean_data(dialogue)

  # token
  inputs = tokenizer(
      dialogue,
      max_length=512,
      padding = "max_length",
      truncation=True,
      return_tensors="pt"

  ).to(device)
  
  
  

  # generate the summary => token ids
  with torch.no_grad():
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        # min_length=30,
        num_beams = 6,
        length_penalty=1.2,
        early_stopping = True,
    )
    # token ids convert to text => decoding

  summary = tokenizer.decode(targets[0], skip_special_tokens=True)
  return summary

# Api end points 

@app.post("/summarize/")
async def summarize(dialogue_input:DialogueInput):
   summary = summarize_dialogue(dialogue_input.dialogue)
   return {"summary":summary}

@app.get("/", response_class= HTMLResponse)
async def   home(request:Request):
   return templates.TemplateResponse(
    request=request,
    name="summarizecode.html",
    context={"request": request}
)