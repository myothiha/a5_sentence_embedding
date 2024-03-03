from flask import Flask, render_template, request
import numpy as np
import pickle
import torch
from models.classes import MyBertTokenizer, BERT, calculate_similarity, meta
from library.utils import get_text_transform, mmtokenizer
import torchtext

app = Flask(__name__)
app.config['TIMEOUT'] = 600

# Load Model

# arg_path = 'models/bert.args'
# meta = pickle.load(open(arg_path, 'rb'))
word2id = meta['word2id']
tokenizer = MyBertTokenizer(word2id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BERT()
model.load_state_dict(torch.load("models/sentence_bert.pt", map_location=torch.device('cpu') ))
model.to(device)
model.eval()

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/compare', methods=['POST'])
def compare():

    # get prompt from HTML form.
    sent1 = request.form.get('sent1')
    sent2 = request.form.get('sent2')

    print('Input:', sent1, sent2)
    similarity = calculate_similarity(model, tokenizer, sent1, sent2, device)

    return render_template('index.html', result=similarity, old_sent1=sent1, old_sent2=sent2)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)