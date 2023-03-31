import os
from io import BytesIO

import openai
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding
from PyPDF2 import PdfReader

load_dotenv()


def extract_text(pdf):
    print("Parsing paper")
    number_of_pages = len(pdf.pages)
    print(f"Total number of pages: {number_of_pages}")
    paper_text = []
    for i in range(number_of_pages):
        page = pdf.pages[i]
        page_text = []

        def visitor_body(text, cm, tm, fontDict, fontSize):
            x = tm[4]
            y = tm[5]
            # ignore header/footer
            if (y > 50 and y < 720) and (len(text.strip()) > 1):
                page_text.append({
                    'fontsize': fontSize,
                    'text': text.strip().replace('\x03', ''),
                    'x': x,
                    'y': y
                })

        _ = page.extract_text(visitor_text=visitor_body)

        blob_font_size = None
        blob_text = ''
        processed_text = []

        for t in page_text:
            if t['fontsize'] == blob_font_size:
                blob_text += f" {t['text']}"
                if len(blob_text) >= 2000:
                    processed_text.append({
                        'fontsize': blob_font_size,
                        'text': blob_text,
                        'page': i
                    })
                    blob_font_size = None
                    blob_text = ''
            else:
                if blob_font_size is not None and len(blob_text) >= 1:
                    processed_text.append({
                        'fontsize': blob_font_size,
                        'text': blob_text,
                        'page': i
                    })
                blob_font_size = t['fontsize']
                blob_text = t['text']
            paper_text += processed_text
    print("Done parsing paper")
    # print(paper_text)
    return paper_text


def create_df(pdf):
    print('Creating dataframe')
    filtered_pdf = []
    for row in pdf:
        if len(row['text']) < 30:
            continue
        filtered_pdf.append(row)
    df = pd.DataFrame(filtered_pdf)
    # print(df.shape)
    # remove elements with identical df[text] and df[page] values
    df = df.drop_duplicates(subset=['text', 'page'], keep='first')
    df['length'] = df['text'].apply(lambda x: len(x))
    print('Done creating dataframe')
    return df


def embeddings(df):
    print('Calculating embeddings')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    embedding_model = "text-embedding-ada-002"
    embeddings = df.text.apply(
        [lambda x: get_embedding(x, engine=embedding_model)])
    df["embeddings"] = embeddings
    print('Done calculating embeddings')
    return df


file = open('book.pdf', 'rb').read()
pdf = PdfReader(BytesIO(file))
text = extract_text(pdf)

df = create_df(text)
df = embeddings(df)

embeddings_json = df.to_json('embeddings.json')
