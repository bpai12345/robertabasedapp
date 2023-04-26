import streamlit as st
import easyocr
import numpy as np
from haystack import Document, Pipeline
from haystack.nodes import FARMReader
from PIL import ImageDraw, Image
import fitz

# Initialize extracted data as an empty dictionary
extracted_data = {}

# Set the threshold for the answer score
threshold_high = 0.7

# Function to draw bounding boxes on image
def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

# Function to get the answer from the result
def get_answer(result):
    if result['answers'][0].score >= threshold_high and result['answers'][0].score <= 1:
        return result['answers'][0].answer
    else:
        return '-'

# Function to extract data from the uploaded PDF file
def extraction(file, existing_data):
    easyreader = easyocr.Reader(['en'], gpu=False, detector=True)
    new_reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        page = doc.load_page(0)  # number of page
        pix = page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    bounds = easyreader.readtext(np.array(image), min_size=0, slope_ths=0.2, ycenter_ths=0.5, height_ths=0.5, y_ths=0.3, low_text=0.5, text_threshold=0.7, width_ths=0.8, paragraph=True, decoder='beamsearch', beamWidth=10)
    context = '\n'.join([b[1] for b in bounds])
    p = Pipeline()
    p.add_node(component=new_reader, name="reader", inputs=["Query"])
    #queries = ["invoice_number?","invoice date?","Seller name?","Address?","Seller Phone number?","Seller email Id?","Seller Tax/GST/VAT number?","Seller website?","Buyer billing name?","Buyer shipping address?","Buyer phone number?","Buyer email Id?","Buyer Tax/GST/VAT number?","Sales tax/GST percentage?","Gross total?","Net amount?","description"]
    queries=["invoice number?","invoice date?","Seller name?","Address?","Buyer billing name?","Buyer shipping address?","Net amount","Due date?","Terms?"]
    results = [p.run(query=q, documents=[Document(content=context)]) for q in queries]
    myData = {q: get_answer(r) for q, r in zip(queries, results)}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Uploaded Invoice')
        st.image(image, caption='Uploaded PDF', use_column_width=True)

    with col2:
        st.subheader('Extracted Details')
        new_data = {}
        with st.form("details_form"):
            for k, v in myData.items():
                if v != "-":
                    if k in ["invoice number?","invoice date?","Seller name?","Address?","Buyer billing name?","Buyer shipping address?","Net amount","Due date?","Terms?"]:
                        # Use the existing data if available, otherwise use the extracted data
                        v = st.text_input(k, value=existing_data.get(k, v))
                        new_data[k] = v
                    else:
                        new_data[k] = v        
            if st.form_submit_button("Save Changes"):
                st.success("Changes saved successfully!")
         
    return bounds, image


st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center; color:orange;'>INVOICE PARSING</h1>", unsafe_allow_html=True)
file = st.file_uploader("Upload a PDF file", type=['pdf'])
if file is not None:
    with st.spinner("Extracting Invoice Data..."):
        bounds, image = extraction(file, {})

