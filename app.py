
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF parsing
import io

st.set_page_config(page_title="Fund Report Analyzer", layout="wide")

st.title("📊 Fund Report Analyzer")
st.write("Upload a PDF or Excel file containing fund performance reports to extract and analyze data.")

uploaded_file = st.file_uploader("Upload file", type=["pdf", "xlsx", "xls"])

if uploaded_file:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write("File uploaded:", file_details)

    if uploaded_file.type == "application/pdf":
        # Use PyMuPDF to extract text from PDF
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

        st.subheader("📄 Extracted Text from PDF")
        st.text_area("Text Output", text, height=400)
    
    elif "excel" in uploaded_file.type or uploaded_file.name.endswith((".xls", ".xlsx")):
        # Read Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        st.subheader("📈 Excel Data Preview")
        st.dataframe(df)

        st.markdown("### Summary Statistics")
        st.write(df.describe())
else:
    st.info("Please upload a PDF or Excel file to get started.")
