#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import pandas as pd
import PyPDF2
import matplotlib.pyplot as plt
import re
from io import BytesIO
from openpyxl import load_workbook
from difflib import get_close_matches
import matplotlib.backends.backend_pdf
import base64

# Page configuration
st.set_page_config(
    page_title="Fund Report Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Fund Report Analyzer Demo")
st.write("Upload your fund return files (PDF or Excel) to process them and generate a consolidated report.")

# Fuzzy Column Matching for Excel Files
def match_column(possible_names, available_columns):
    """
    Performs fuzzy matching for column names.
    """
    for name in possible_names:
        match = get_close_matches(name.lower(), available_columns, n=1, cutoff=0.6)
        if match:
            return match[0]
    return None

# Data Extraction
def parse_aum(aum_string):
    """
    Parses AUM strings into a consistent numerical format (in Millions USD).
    Handles 'B' for billions, commas, and non-numeric characters.
    """
    if not isinstance(aum_string, str):
        return float(aum_string)

    aum_string = aum_string.strip().lower()
    num_str = re.sub(r'[^\d.]', '', aum_string)
    if not num_str:
        return None

    num = float(num_str)

    if 'b' in aum_string:
        return num * 1000 # Convert billions to millions for consistency
    return num

def clean_text_field(text):
    """
    Cleans extracted text fields to remove noise.
    """
    if isinstance(text, str):
        # Remove any leading/trailing special characters like '*', '-', ':'
        text = re.sub(r'^[^\w\s/]+|[^\w\s/]+$', '', text)
        # Replace multiple newlines/spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove specific known noise patterns if they appear within names
        text = text.replace('**Weekly Performance Update**', '').replace('Strategy | Fund | Week Return | Assets | ||', '')
        # Remove common artifacts like '**', '|', '--' that might still be present
        text = re.sub(r'[\*\-|]+', '', text)
        text = text.strip()
    return text

def extract_pdf_data(file_content, filename):
    """
    Extracts fund data from PDF files using various regex patterns for different formats.
    """
    st.write(f"--- Processing PDF: {filename} ---")
    data = []
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # --- Specific pre-processing for __Weekly Performance Update__.pdf ---
        if "__Weekly Performance Update__.pdf" in filename:
            # Combine lines split by the PyPDF2-specific `","` artifact
            text = text.replace('"\n","', ' ') # Merges lines like "...\n", "..." -> "... ..."
            text = text.replace('"\n', ' ')    # Removes trailing " and newline from a segment
            text = text.replace('"', '')       # Remove any remaining quotes
            text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with a single space
            text = text.replace('\n', ' ')

    except Exception as e:
        st.error(f"Error reading PDF file {filename}: {e}")
        return pd.DataFrame()

    crest_pattern = re.compile(
        r'-\s*\*\*(?P<fund_name>[^*]+?)\*\*\s*:\s*(?P<return_val>[\d\.\-]+)%\s*'
        r'(?:\(AUM:\s*(?P<aum_val>[\d\.\,]+\s*[MB]?)\))?\s*\|\s*Strat:\s*(?P<strategy_val>[\w\-]+)',
        re.IGNORECASE | re.MULTILINE
    )

    pattern_report1 = re.compile(
        r'Fund\s+Name\s*:\s*(?P<fund_name>.*?)\s*\n+'
        r'Return\s+\(%\)\s*:\s*(?P<return_val>[\d\.\-]+)\s*\n+'
        r'AUM\s*:\s*(?P<aum_val>[\d\.\,]+\s*[MB]?)\s*\n+'
        r'Category\s*:\s*(?P<strategy_val>.*?)\s*\n+',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    weekly_perf_pattern = re.compile(
        r'(?P<strategy>[^|]+?)\s*\|\s*'
        r'(?P<fund_name>[^|]+?)\s*\|\s*'
        r'(?P<return_val>[\+\-]?[\d\.]+%)?\s*\|\s*'
        r'(?P<aum_val>[\d\.\,]+\s*[MB]?)?\s*\|\s*',
        re.IGNORECASE | re.DOTALL # DOTALL to match across lines if text wasn't fully flattened
    )

    # General pattern for documents with "Fund Name : X% (AUM: Y) | Strat: Z"
    pattern = re.compile(
        r"([\w\s\*\-]+?)\s*:\s*"
        r"([\d\.\-]+)%\s*"
        r"(?:\(AUM:\s*([\d\.\,]+\s*[MB]?)\))?\s*"
        r"\|\s*Strat:\s*([\w\-]+)",
        re.IGNORECASE | re.MULTILINE
    )

    # General pattern for documents with "Fund Name | X% | Y | Z"
    pipe_pattern = re.compile(
        r"([\w\s]+)\s*\|\s*"
        r"([\d\.\-]+)%\s*\|\s*"
        r"([\d\.\,]+)\s*\|\s*"
        r"([\w\-]+)",
        re.IGNORECASE | re.MULTILINE
    )

    # Improved pipe-separated pattern for standard reports
    standard_report_pattern = re.compile(
        r'([A-Za-z\s]+?)\s*\|\s*'
        r'([\+\-]?[\d\.]+)%\s*\|\s*'
        r'([\d\.\,]+)\s*\|\s*'
        r'([A-Za-z\s/\-]+)',
        re.IGNORECASE | re.MULTILINE
    )

    extracted_count = 0

    # Try Crest.pdf specific pattern first
    matches = crest_pattern.finditer(text)
    for match in matches:
        try:
            fund_name = clean_text_field(match.group('fund_name'))
            ret = float(match.group('return_val')) / 100
            aum_str = match.group('aum_val')
            aum = parse_aum(aum_str) if aum_str else None
            strategy = clean_text_field(match.group('strategy_val'))

            data.append({
                "fund_name": fund_name,
                "return": ret,
                "aum": aum,
                "strategy": strategy
            })
            extracted_count += 1
        except Exception as e:
            st.warning(f"Could not parse Crest format match: {match.groups()} - Error: {e}")

    # Try __Weekly Performance Update__.pdf specific pattern next
    if extracted_count == 0 or "__Weekly Performance Update__.pdf" in filename:
        matches = weekly_perf_pattern.finditer(text)
        for match in matches:
            try:
                strategy = clean_text_field(match.group('strategy'))
                fund_name = clean_text_field(match.group('fund_name'))

                # Skip common header/footer lines explicitly
                if any(phrase in fund_name.lower() for phrase in ["strategy | fund", "week return", "assets"]):
                    continue
                if "weekly performance update" in fund_name.lower():
                    continue

                ret_val = match.group('return_val')
                ret = float(ret_val.replace('%', '')) / 100 if ret_val else None
                aum_str = match.group('aum_val')
                aum = parse_aum(aum_str) if aum_str else None

                data.append({
                    "fund_name": fund_name,
                    "return": ret,
                    "aum": aum,
                    "strategy": strategy
                })
                extracted_count += 1
            except Exception as e:
                st.warning(f"Could not parse Weekly Performance format match: {match.groups()} - Error: {e}")

    # Try Report1.pdf specific pattern if previous patterns didn't find anything
    if extracted_count == 0:
        matches = pattern_report1.finditer(text)
        for match in matches:
            try:
                fund_name = clean_text_field(match.group('fund_name'))
                ret = float(match.group('return_val')) / 100
                aum_str = match.group('aum_val')
                aum = parse_aum(aum_str) if aum_str else None
                strategy = clean_text_field(match.group('strategy_val'))

                data.append({
                    "fund_name": fund_name,
                    "return": ret,
                    "aum": aum,
                    "strategy": strategy
                })
                extracted_count += 1
            except Exception as e:
                st.warning(f"Could not parse Report1-format match: {match.groups()} - Error: {e}")

    # Fallback to general pattern if still no data extracted
    if extracted_count == 0:
        matches = pattern.finditer(text)
        for match in matches:
            try:
                raw_name = match.group(1)
                fund_name = clean_text_field(raw_name)

                # Skip common header/footer lines explicitly
                if any(phrase in fund_name.lower() for phrase in ["strategy | fund", "week return", "assets", "fund name", "return", "aum", "category"]):
                    continue
                if "weekly performance update" in fund_name.lower():
                    continue

                ret = float(match.group(2).replace('%', '')) / 100
                aum_str = match.group(3)
                aum = parse_aum(aum_str) if aum_str else None
                strategy = clean_text_field(match.group(4))

                data.append({
                    "fund_name": fund_name,
                    "return": ret,
                    "aum": aum,
                    "strategy": strategy
                })
                extracted_count +=1
            except Exception as e:
                st.warning(f"Could not parse general format match: {match.groups()} - Error: {e}")

    # Try the improved standard report pattern for most PDFs
    if extracted_count == 0:
        matches = standard_report_pattern.finditer(text)
        for match in matches:
            try:
                fund_name = clean_text_field(match.group(1))
                
                # Skip headers and common noise
                if any(phrase in fund_name.lower() for phrase in [
                    "fund name", "strategy", "weekly fund", "returns report", 
                    "------", "data as of", "weekly performance"
                ]):
                    continue
                
                ret = float(match.group(2)) / 100
                aum = parse_aum(match.group(3))
                strategy = clean_text_field(match.group(4))
                
                data.append({
                    "fund_name": fund_name,
                    "return": ret,
                    "aum": aum,
                    "strategy": strategy
                })
                extracted_count += 1
            except Exception as e:
                st.warning(f"Could not parse standard report match: {match.groups()} - Error: {e}")

    # Fallback to pipe-separated pattern if still no data extracted
    if extracted_count == 0:
        matches = pipe_pattern.finditer(text)
        for match in matches:
            try:
                raw_name = match.group(1)
                fund_name = clean_text_field(raw_name)

                # Skip common header/footer lines explicitly
                if any(phrase in fund_name.lower() for phrase in ["strategy | fund", "week return", "assets", "fund name", "return", "aum", "category"]):
                    continue
                if "weekly performance update" in fund_name.lower():
                    continue

                ret = float(match.group(2).replace('%', '')) / 100
                aum = parse_aum(match.group(3))
                strategy = clean_text_field(match.group(4))
                data.append({
                    "fund_name": fund_name,
                    "return": ret,
                    "aum": aum,
                    "strategy": strategy
                })
                extracted_count += 1
            except Exception as e:
                st.warning(f"Could not parse pipe-format match: {match.groups()} - Error: {e}")

    st.write(f"Extracted {len(data)} records from {filename}")
    return pd.DataFrame(data)

def extract_excel_data(file_content, filename):
    """
    Extracts fund data from Excel files.
    """
    st.write(f"--- Processing Excel: {filename} ---")
    df = pd.read_excel(BytesIO(file_content))
    df.columns = [col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "") for col in df.columns]
    cols = list(df.columns)
    
    st.write(f"Found columns: {cols}")

    fund_col = match_column(["fund_name", "fund"], cols)
    ret_col = match_column(["weekly_return", "weekly_return_", "return", "performance"], cols)
    aum_col = match_column(["aum_m_usd", "aum", "net_assets", "assets"], cols)
    strat_col = match_column(["strategy", "strat", "approach"], cols)

    st.write(f"Matched columns - Fund: {fund_col}, Return: {ret_col}, AUM: {aum_col}, Strategy: {strat_col}")

    if not all([fund_col, ret_col, strat_col]):
        raise ValueError(f"Missing required columns (Fund, Return, Strategy) in {filename}")

    new_df = pd.DataFrame()
    new_df["fund_name"] = df[fund_col].apply(clean_text_field)
    new_df["return"] = pd.to_numeric(df[ret_col], errors='coerce') / 100
    new_df["strategy"] = df[strat_col].apply(clean_text_field)
    new_df["aum"] = df[aum_col].apply(parse_aum) if aum_col else None

    new_df.dropna(subset=['fund_name', 'return', 'strategy'], inplace=True)

    st.write(f"Extracted {len(new_df)} records from {filename}")
    return new_df[["fund_name", "return", "aum", "strategy"]]

def standardize_strategy_names(df):
    """
    Standardizes strategy names in the DataFrame.
    """
    df['strategy'] = df['strategy'].replace({
        'L/S Equity': 'Long/Short Equity',
        'L/S Eq': 'Long/Short Equity',
        'L/S': 'Long/Short Equity',
        'Long/Short Eq': 'Long/Short Equity',
        'Fixed Income Arb': 'Fixed Income Arbitrage',
        'Fixed Income Arbitrage': 'Fixed Income Arbitrage',
        'Global Macro': 'Global Macro',
        'EventDriven': 'Event-Driven',
        'Credit': 'Credit',
        'Multi-Strategy': 'Multi-Strategy',
        'Quant': 'Quantitative',
        'Vol': 'Volatility',
        'Vol Arb': 'Volatility Arbitrage',
        'Commodity': 'Commodities',
        'Commodity Trading': 'Commodities'
    }).str.strip() # Remove any leading/trailing whitespace after standardization
    return df

def create_download_link(df, filename, label):
    """Create a download link for a DataFrame as Excel file"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='FundData')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{label}</a>'
    return href

def create_pdf_download_link(pdf_bytes, filename, label):
    """Create a download link for PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{label}</a>'
    return href

# File uploader
uploaded_files = st.file_uploader(
    "Choose files", 
    accept_multiple_files=True,
    type=['pdf', 'xlsx', 'xls'],
    help="Upload PDF or Excel files containing fund return data"
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} files")
    
    # Process files
    all_data = []
    
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read()
        filename = uploaded_file.name
        
        try:
            if filename.lower().endswith(".pdf"):
                df = extract_pdf_data(file_content, filename)
            elif filename.lower().endswith((".xlsx", ".xls")):
                df = extract_excel_data(file_content, filename)
            else:
                st.warning(f"Skipping unsupported file type: {filename}")
                continue

            if not df.empty:
                all_data.append(df)
        except Exception as e:
            st.error(f"!!! Failed to process {filename}: {e} !!!")

    # Display, Analyze, and Export
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Standardize strategy names
        combined_df = standardize_strategy_names(combined_df)

        # Calculate Net Return in USD if AUM and Return data are available
        if "aum" in combined_df.columns and "return" in combined_df.columns:
            combined_df["net_return_usd"] = combined_df["return"] * combined_df["aum"]

        st.subheader("📋 Combined Fund Data")
        st.dataframe(combined_df, use_container_width=True)

        st.subheader("📊 Analysis & Visualizations")

        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot 1: Average Return by Fund
            if not combined_df.empty:
                st.subheader("Average Return by Fund")
                avg_returns = combined_df.groupby("fund_name")["return"].mean().sort_values()
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(range(len(avg_returns)), avg_returns.values)
                ax.set_yticks(range(len(avg_returns)))
                ax.set_yticklabels(avg_returns.index)
                ax.set_xlabel("Average Return")
                ax.set_title("Average Return by Fund")
                
                # Add labels to bars
                for i, v in enumerate(avg_returns.values):
                    ax.text(v + 0.0005, i, f"{v:.2%}", color='black', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Plot 3: Average Return by Strategy
            if not combined_df.empty:
                st.subheader("Average Return by Strategy")
                avg_ret_by_strategy = combined_df.groupby("strategy")["return"].mean().sort_values()
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(range(len(avg_ret_by_strategy)), avg_ret_by_strategy.values, color="green")
                ax.set_yticks(range(len(avg_ret_by_strategy)))
                ax.set_yticklabels(avg_ret_by_strategy.index)
                ax.set_xlabel("Average Return")
                ax.set_title("Average Return by Strategy")
                
                # Add labels to bars
                for i, v in enumerate(avg_ret_by_strategy.values):
                    ax.text(v + 0.0005, i, f"{v:.2%}", color='black', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with col2:
            # Plot 2: Total AUM by Strategy
            if "aum" in combined_df.columns and not combined_df["aum"].isnull().all():
                st.subheader("Total AUM by Strategy")
                aum_by_strategy = combined_df.dropna(subset=["aum"]).groupby("strategy")["aum"].sum().sort_values()
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(range(len(aum_by_strategy)), aum_by_strategy.values, color="orange")
                ax.set_yticks(range(len(aum_by_strategy)))
                ax.set_yticklabels(aum_by_strategy.index)
                ax.set_xlabel("Total AUM (M USD)")
                ax.set_title("Total AUM by Strategy")
                
                # Add labels to bars
                for i, v in enumerate(aum_by_strategy.values):
                    ax.text(v + 10, i, f"${v:,.0f}M", color='black', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Plot 4: Total AUM by Fund
            if "aum" in combined_df.columns and not combined_df["aum"].isnull().all():
                st.subheader("Total AUM by Fund")
                aum_by_fund = combined_df.dropna(subset=["aum"]).groupby("fund_name")["aum"].sum().sort_values()
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(range(len(aum_by_fund)), aum_by_fund.values, color="purple")
                ax.set_yticks(range(len(aum_by_fund)))
                ax.set_yticklabels(aum_by_fund.index)
                ax.set_xlabel("Total AUM (M USD)")
                ax.set_title("Total AUM by Fund")
                
                # Add labels to bars
                for i, v in enumerate(aum_by_fund.values):
                    ax.text(v + 10, i, f"${v:,.0f}M", color='black', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Summary Statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Funds", combined_df['fund_name'].nunique())
            
        with col2:
            st.metric("Total Strategies", combined_df['strategy'].nunique())
            
        with col3:
            if "aum" in combined_df.columns and not combined_df['aum'].isnull().all():
                total_aum = combined_df['aum'].sum()
                st.metric("Total AUM", f"${total_aum:,.0f}M")

        # Key Insights
        st.subheader("🔍 Key Insights")
        
        if not combined_df.empty:
            avg_ret_by_strategy = combined_df.groupby("strategy")["return"].mean().sort_values(ascending=False)
            top_strategy = avg_ret_by_strategy.index[0]
            top_return = avg_ret_by_strategy.iloc[0]
            
            st.write(f"**Top Performing Strategy by Average Return:** {top_strategy} ({top_return:.2%})")

        # Net Return Summary
        if "net_return_usd" in combined_df.columns and not combined_df["net_return_usd"].isnull().all():
            st.subheader("Net Return Summary (USD Millions)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**By Strategy:**")
                net_return_by_strategy = combined_df.groupby("strategy")["net_return_usd"].sum().sort_values(ascending=False)
                for strategy, net_ret in net_return_by_strategy.items():
                    st.write(f"- {strategy}: ${net_ret:,.2f}M")
            
            with col2:
                st.write("**By Fund:**")
                net_return_by_fund = combined_df.groupby("fund_name")["net_return_usd"].sum().sort_values(ascending=False)
                for fund, net_ret in net_return_by_fund.head(10).items():  # Show top 10
                    st.write(f"- {fund}: ${net_ret:,.2f}M")

        # Download Options
        st.subheader("⬇️ Download Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download
            excel_link = create_download_link(combined_df, "combined_fund_report.xlsx", "📊 Download Excel Report")
            st.markdown(excel_link, unsafe_allow_html=True)
        
        with col2:
            # Generate PDF report
            if st.button("📄 Generate PDF Report"):
                # Create PDF with all charts
                pdf_buffer = BytesIO()
                pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_buffer)
                
                # Recreate all plots for PDF
                plots_data = [
                    ("Average Return by Fund", combined_df.groupby("fund_name")["return"].mean().sort_values(), "return", "Fund Name", "Average Return"),
                    ("Average Return by Strategy", combined_df.groupby("strategy")["return"].mean().sort_values(), "return", "Strategy", "Average Return")
                ]
                
                if "aum" in combined_df.columns and not combined_df["aum"].isnull().all():
                    plots_data.extend([
                        ("Total AUM by Strategy", combined_df.dropna(subset=["aum"]).groupby("strategy")["aum"].sum().sort_values(), "aum", "Strategy", "Total AUM (M USD)"),
                        ("Total AUM by Fund", combined_df.dropna(subset=["aum"]).groupby("fund_name")["aum"].sum().sort_values(), "aum", "Fund Name", "Total AUM (M USD)")
                    ])
                
                for title, data, data_type, ylabel, xlabel in plots_data:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = ax.barh(range(len(data)), data.values)
                    ax.set_yticks(range(len(data)))
                    ax.set_yticklabels(data.index)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    
                    # Add labels to bars
                    for i, v in enumerate(data.values):
                        if data_type == "return":
                            ax.text(v + 0.0005, i, f"{v:.2%}", color='black', va='center')
                        else:
                            ax.text(v + 10, i, f"${v:,.0f}M", color='black', va='center')
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # Summary page
                fig_text, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                summary_text = [
                    "Final Report Summary",
                    "--------------------------------",
                    f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                    f"Total Funds Analyzed: {combined_df['fund_name'].nunique()}",
                    f"Total Strategies Analyzed: {combined_df['strategy'].nunique()}",
                    "",
                    "**Key Insights:**"
                ]
                
                if not combined_df.empty:
                    avg_ret_by_strategy = combined_df.groupby("strategy")["return"].mean().sort_values(ascending=False)
                    top_strategy = avg_ret_by_strategy.index[0]
                    top_return = avg_ret_by_strategy.iloc[0]
                    summary_text.append(f"Top Performing Strategy by Avg Return: '{top_strategy}' ({top_return:.2%})")
                
                if "aum" in combined_df.columns and not combined_df['aum'].isnull().all():
                    total_aum = combined_df['aum'].sum()
                    summary_text.append(f"Total AUM Across All Funds: ${total_aum:,.0f}M")
                
                # Add text to the figure
                y_pos = 0.9
                for line in summary_text:
                    ax.text(0.05, y_pos, line, fontsize=12, ha='left', va='top')
                    y_pos -= 0.05
                
                plt.tight_layout()
                pdf.savefig(fig_text)
                plt.close(fig_text)
                
                pdf.close()
                pdf_bytes = pdf_buffer.getvalue()
                pdf_buffer.close()
                
                # Create download link
                pdf_link = create_pdf_download_link(pdf_bytes, "summary_report.pdf", "📄 Download PDF Report")
                st.markdown(pdf_link, unsafe_allow_html=True)
                st.success("PDF report generated successfully!")

    else:
        st.warning("No valid data could be extracted from the uploaded files.")
else:
    st.info("Please upload PDF or Excel files to get started.")
