import pandas as pd
import streamlit as st
import gensim
import nltk
import base64
import time
import re
from tqdm import tqdm
import string
tqdm.pandas()
st.set_option('deprecation.showfileUploaderEncoding', False)
FILE_TYPES = ["csv", "xlsx"]
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopwords = nltk.corpus.stopwords.words('english')


def nltkPreprocessing(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = tokenizer.tokenize(text)
    text = [w for w in text if w not in stopwords]
    text = ' '.join(text)
    return text


def gensimPreprocessing(text):
    result = []
    for token in gensim.utils.simple_preprocess(str(text)):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return ' '.join(result)


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()

    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download CSV</a>'


if __name__ == "__main__":
    st.title("Text Preprocessing for NLP")
    st.text("<= 25 MB Suggested. Max = 35 MB")
    file_type = st.selectbox("File Type - ", ('csv', 'xlxs'))
    file = st.file_uploader("Upload file", file_type)
    show_file = st.empty()
    if not file:
        time.sleep(0.01)
        show_file.info("Please upload a file of type: " +
                       ", ".join(FILE_TYPES))

    if file:

        data = 0
        if file_type == 'csv':
            data = pd.read_csv(file)
            st.write("Shape", data.shape)
        else:
            data = pd.read_excel(file)
            st.write("Shape", data.shape)
        file.close()

        cols = data.columns
        column = st.selectbox("Which column you want to preprocess?",
                              (cols))
        prepType = st.selectbox("Which type Preprocessing ?",
                                ("nltk", "gensim"))

        if (st.button("Submit", )):

            latest_iteration = st.empty()
            bar = st.progress(0)
            length = max(data.shape)
            if prepType == "nltk":
                results = []
                for idx, text in enumerate(data[column]):
                    results.append(nltkPreprocessing(text))
                    latest_iteration.text(f'Row - {idx + 1}')
                    bar.progress((idx+1)/length)
                pred = pd.DataFrame(results, columns=[f'{column}_prep'])
                data = pd.concat([data, pred], axis=1)

            else:
                results = []
                for idx, text in enumerate(data[column]):
                    results.append(gensimPreprocessing(text))
                    latest_iteration.text(f'Row - {idx+1}')
                    bar.progress((idx + 1)/length)
                pred = pd.DataFrame(results, columns=[f'{column}_prep'])
                data = pd.concat([data, pred], axis=1)

            st.markdown(get_table_download_link(data), unsafe_allow_html=True)

            st.text("If Failed! Final file size exceeded 50 MB")

        if (st.button("Get Code", )):
            st.code('''
#nltk.preprocessing
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopwords = nltk.corpus.stopwords.words('english')

def nltkPreprocessing(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = tokenizer.tokenize(text)
    text = [w for w in text if w not in stopwords]
    text = ' '.join(text)
    return text

#gensim.preprocessing
def gensimPreprocessing(text):
    result = []
    for token in gensim.utils.simple_preprocess(str(text)):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return ' '.join(result) 
''', language="python")

