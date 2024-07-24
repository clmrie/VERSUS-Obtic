
# libraries
from flask import Flask, request, jsonify, render_template
from flask import make_response
from flask_cors import CORS

"""
from code_folder.code import call
from code_folder.code import get_indices
from code_folder.code import replace_sentences_html
"""

from code_folder.code2 import call
from code_folder.code2 import get_indices
from code_folder.code2 import replace_sentences_html

import pypdf

app = Flask(__name__)
CORS(app)

# redirects to main page HTML
@app.route('/')
def index():
    return render_template('index.html')

# function to execute python code with what is sent from script.js
@app.route('/process', methods=['GET', 'POST'])
def compare_texts():
    data = request.json
    text1Text = data['text1Text']
    text2Text = data['text2Text']
    text1HTML = data['text1HTML']
    text2HTML = data['text2HTML']
    method = data['method']
    submethods = data['submethods']
    slidingValue = data['slidingValue']
    embedding_model = data['embedding'] 
    top_quantile =float(data['sliderConfidence'])
    precision_label = data['precisionLabel']
    textProcess = data['textProcess']
    ngramsInput = data['ngramsInput']
    nb_sentences_value = data['nb_sentences_value']
    
    text1Text = text1Text.replace('\n', '\\n')
    text2Text = text2Text.replace('\n', '\\n')

    df_comp = call(text1Text, text2Text, method, slidingValue, submethods, embedding_model, top_quantile, precision_label,
                   textProcess, ngramsInput, nb_sentences_value)
    _, _, _, sent_list_1, sent_list_2 = get_indices(df_comp, text1HTML, text2HTML)

    # colors for highlighting
    colors_list = [
    'blue',
    'teal',
    'maroon',
    'indigo',
    'olive',
    'sienna',
    'orange',
    'black',
    'tomato',
    'plum',
    'coral',
    'indigo',
    'green',
    'purple',
    'red',
    'cyan',
    'magenta',
    'lime',
    'pink',
    'azure',
    'beige',
    'brown',
    'gold',
    'gray'
    ]
    #class_text_1 =  get_indices_highlighted(text1Text, text2Text, highlight_ranges_text1, highlight_ranges_text2)
    text_1, text_2 = replace_sentences_html(text1Text, text2Text,
                                            sent_list_1, sent_list_2,
                                            colors_list)

    return jsonify({'text_1': text_1,
                    'text_2': text_2,
                    'sent_list_1': sent_list_1,
                    'sent_list_2': sent_list_2})




@app.route('/extract_pdf_text', methods=['GET', 'POST'])
def extract_pdf_text():
    if 'pdfFile' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400
    
    def extract_text_from_pdf(pdf_file):
        text = ""
        pdf_reader = pypdf.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

    pdf_file = request.files['pdfFile']
    text = extract_text_from_pdf(pdf_file)

    return jsonify({'pdfText': text})

if __name__ == '__main__':  
    app.run(debug=True)

