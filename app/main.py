import os
import numpy as np
from fastapi import FastAPI
from transformers import DistilBertTokenizerFast, DistilBertModel
from NLPModel import GenericBertMode
import torch

app = FastAPI()

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")


model = GenericBertMode.BERT_Arch(distilbert)
# push model to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# load weights

path = os.path.join(os.path.curdir,'NLPModel/','saved_weights_distilbert.pt')
model.load_state_dict(torch.load(path,map_location=device))

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/inference")
async def model_prediction(message: str):
    inputTest = list()
    inputTest.append(message)
    inputTokenized = tokenizer(
        inputTest,
        max_length=25,
        padding='max_length',
        truncation=True
    )

    text_input_ids, text_attention_max = inputTokenized["input_ids"], inputTokenized["attention_mask"]

    text_seq = torch.tensor(text_input_ids).to(device)
    text_mask = torch.tensor(text_attention_max).to(device)

    with torch.no_grad():
        pred = model(text_seq, text_mask)
        pred = pred.detach().cpu().numpy()

    pred = np.argmax(pred, axis=1)
    res = "This message has been classified has spam" if pred == 1 else "This message has been classified as normal"
    value = 1 if pred == 1 else 0
    return {'input':message, 'classification':res,'value':value}
