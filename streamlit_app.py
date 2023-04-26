import streamlit as st
import transformers
import torch
import pdfplumber

st.title("DocumentParser")


def load_model():
    model = transformers.AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    return model, tokenizer

model, tokenizer = load_model()

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    st.write(text)

    question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        st.write("Answer:", answer)
