from flask import Flask, request, jsonify
import os
import gspread
from google.oauth2.service_account import Credentials
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import json
import torch



app = Flask(__name__)


logging.basicConfig(level=logging.DEBUG)

# Google Sheets Setup
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("/home/bonnopitom/mysite/ee-csebrurhasinmanjare3434-0f1939796706.json", scopes=SCOPES) #chnage this credentials path too
gc = gspread.authorize(creds)
spreadsheet = gc.open("Leads_Management").sheet1


tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

#change_this_model_path_please
model = torch.load("/home/bonnopitom/mysite/bert_model.pth")

    

def predict_label(text):
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
    
   
    return "Interested" if predictions.item() == 1 else "Not Interested"

@app.route('/reply', methods=['POST'])
def handle_reply():
    try:
        
        reply_data = request.json

        
        app.logger.debug(f"Webhook Response: {json.dumps(reply_data, indent=2)}")

        
        if not isinstance(reply_data, list):
            return jsonify({'message': 'Invalid data format'}), 400

        for event in reply_data:
            sender = event.get('email') or event.get('from_email')  
            body = event.get('text', '').strip().lower() 

           
            app.logger.debug(f"Sender: {sender}, Body: {body}")

            
            if not sender:
                app.logger.warning("No sender email found, skipping.")
                continue

            
            try:
                cell = spreadsheet.find(sender, in_column=2)  
                row = cell.row
            except gspread.exceptions.CellNotFound:
                app.logger.warning(f"Email {sender} not found in Google Sheets.")
                continue  

            
            status = predict_label(body)

            
            spreadsheet.update_cell(row, 7, status)
            app.logger.info(f"Updated row {row} for {sender} with status '{status}'")

        return jsonify({'message': 'Reply processed successfully!'}), 200

    except Exception as e:
        app.logger.error(f"Error processing the reply data: {e}")
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000)
