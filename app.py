#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import PyPDF2
import matplotlib.pyplot as plt
import re
from io import BytesIO
from openpyxl import load_workbook
# from google.colab import files
from difflib import get_close_matches

# Create a directory to store uploaded files
os.makedirs("downloads", exist_ok=True)

print("Please upload your fund return files (PDF or Excel).")
print("The script will process them and generate a consolidated report.")
uploaded_files = files.upload()

for filename in uploaded_files:
    # Save uploaded files to the 'downloads' directory
    with open(f"downloads/{filename}", "wb") as f:
        f.write(uploaded_files[filename])
    print(f"Uploaded and saved: {filename}")


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

def extract_pdf_data(filepath):
    """
    Extracts fund data from PDF files using various regex patterns for different formats.
    """
    print(f"--- Processing PDF: {filepath} ---")
    data = []
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            # print("\n--- DEBUG: Full Extracted PDF Text ---\n", text, "\n--- END DEBUG ---\n")

        # --- Specific pre-processing for __Weekly Performance Update__.pdf ---
        if "__Weekly Performance Update__.pdf" in os.path.basename(filepath):
            # Combine lines split by the PyPDF2-specific `","` artifact
            text = text.replace('"\n","', ' ') # Merges lines like "...\n", "..." -> "... ..."
            text = text.replace('"\n', ' ')    # Removes trailing " and newline from a segment
            text = text.replace('"', '')       # Remove any remaining quotes
            text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with a single space
            text = text.replace('\n', ' ')
            # print("\n--- DEBUG: Preprocessed Weekly Performance Text ---\n", text, "\n--- END DEBUG ---\n")

    except Exception as e:
        print(f"Error reading PDF file {filepath}: {e}")
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
            print(f"Could not parse Crest format match: {match.groups()} - Error: {e}")

    # Try __Weekly Performance Update__.pdf specific pattern next
    if extracted_count == 0 or "__Weekly Performance Update__.pdf" in os.path.basename(filepath):
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
                print(f"Could not parse Weekly Performance format match: {match.groups()} - Error: {e}")

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
                print(f"Could not parse Report1-format match: {match.groups()} - Error: {e}")

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
                print(f"Could not parse general format match: {match.groups()} - Error: {e}")

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
                print(f"Could not parse pipe-format match: {match.groups()} - Error: {e}")


    print(f"Extracted {len(data)} records from {filepath}")
    return pd.DataFrame(data)


def extract_excel_data(filepath):
    """
    Extracts fund data from Excel files.
    """
    print(f"--- Processing Excel: {filepath} ---")
    df = pd.read_excel(filepath)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    cols = list(df.columns)

    fund_col = match_column(["fund_name", "fund"], cols)
    ret_col = match_column(["weekly_return_(%)", "weekly_return", "return", "performance"], cols)
    aum_col = match_column(["aum", "aum_(m_usd)", "net_assets", "assets"], cols)
    strat_col = match_column(["strategy", "strat", "approach"], cols)

    if not all([fund_col, ret_col, strat_col]):
        raise ValueError(f"Missing required columns (Fund, Return, Strategy) in {filepath}")

    new_df = pd.DataFrame()
    new_df["fund_name"] = df[fund_col].apply(clean_text_field)
    new_df["return"] = pd.to_numeric(df[ret_col], errors='coerce') / 100
    new_df["strategy"] = df[strat_col].apply(clean_text_field)
    new_df["aum"] = df[aum_col].apply(parse_aum) if aum_col else None

    new_df.dropna(subset=['fund_name', 'return', 'strategy'], inplace=True)

    print(f"Extracted {len(new_df)} records from {filepath}")
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
        'Commodity': 'Commodities'
    }).str.strip() # Remove any leading/trailing whitespace after standardization
    return df

# ----------- 4. Main Parsing Loop -------------
all_data = []

for filename in uploaded_files:
    path = os.path.join("downloads", filename)
    try:
        if filename.lower().endswith(".pdf"):
            df = extract_pdf_data(path)
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = extract_excel_data(path)
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        if not df.empty:
            all_data.append(df)
    except Exception as e:
        print(f"!!! Failed to process {filename}: {e} !!!")

# ----------- 5. Display, Analyze, and Export -------------
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)

    # Standardize strategy names
    combined_df = standardize_strategy_names(combined_df)

    # Calculate Net Return in USD if AUM and Return data are available
    if "aum" in combined_df.columns and "return" in combined_df.columns:
        combined_df["net_return_usd"] = combined_df["return"] * combined_df["aum"]

    print("\n--- Combined Fund Data ---")
    # Using `display` for rich table formatting in Jupyter/Colab
    display(combined_df)

    print("\n--- Analysis & Visualizations ---")

    # Export Combined Data and Generate Reports
    summary_path = "summary_report.pdf"
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(summary_path)

    # Plot 1: Average Return by Fund
    if not combined_df.empty:
        plt.figure(figsize=(10, 8))
        avg_returns = combined_df.groupby("fund_name")["return"].mean().sort_values()
        ax = avg_returns.plot(kind="barh", title="Average Return by Fund")
        plt.xlabel("Average Return")
        plt.ylabel("Fund Name")
        # Add labels to bars
        for i, v in enumerate(avg_returns):
            ax.text(v + 0.0005, i, f"{v:.2%}", color='black', va='center')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Plot 2: Total AUM by Strategy
    if "aum" in combined_df.columns and not combined_df["aum"].isnull().all():
        plt.figure(figsize=(10, 8))
        aum_by_strategy = combined_df.dropna(subset=["aum"]).groupby("strategy")["aum"].sum().sort_values()
        ax = aum_by_strategy.plot(kind="barh", title="Total AUM by Strategy", color="orange")
        plt.xlabel("Total AUM (M USD)")
        plt.ylabel("Strategy")
        # Add labels to bars
        for i, v in enumerate(aum_by_strategy):
            ax.text(v + 10, i, f"${v:,.0f}M", color='black', va='center')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Plot 3: Average Return by Strategy
    if not combined_df.empty:
        plt.figure(figsize=(10, 8))
        avg_ret_by_strategy = combined_df.groupby("strategy")["return"].mean().sort_values()
        ax = avg_ret_by_strategy.plot(kind="barh", title="Average Return by Strategy", color="green")
        plt.xlabel("Average Return")
        plt.ylabel("Strategy")
        # Add labels to bars
        for i, v in enumerate(avg_ret_by_strategy):
            ax.text(v + 0.0005, i, f"{v:.2%}", color='black', va='center')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Plot 4: Total AUM by Fund
    if "aum" in combined_df.columns and not combined_df["aum"].isnull().all():
        plt.figure(figsize=(10, 8))
        aum_by_fund = combined_df.dropna(subset=["aum"]).groupby("fund_name")["aum"].sum().sort_values()
        ax = aum_by_fund.plot(kind="barh", title="Total AUM by Fund", color="purple")
        plt.xlabel("Total AUM (M USD)")
        plt.ylabel("Fund Name")
        # Add labels to bars
        for i, v in enumerate(aum_by_fund):
            ax.text(v + 10, i, f"${v:,.0f}M", color='black', va='center')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Generate summary text page
    fig_text = plt.figure(figsize=(8.5, 11))
    plt.axis('off')

    top_strategy = avg_ret_by_strategy.idxmax() if 'avg_ret_by_strategy' in locals() and not avg_ret_by_strategy.empty else "N/A"
    top_return = avg_ret_by_strategy.max() if 'avg_ret_by_strategy' in locals() and not avg_ret_by_strategy.empty else 0

    summary_text = [
        "Final Report Summary",
        "--------------------------------",
        f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total Funds Analyzed: {combined_df['fund_name'].nunique()}",
        f"Total Strategies Analyzed: {combined_df['strategy'].nunique()}",
        "",
        "**Key Insights:**",
        f"Top Performing Strategy by Avg Return: '{top_strategy}' ({top_return:.2%})",
    ]
    if "aum" in combined_df.columns and not combined_df['aum'].isnull().all():
        total_aum = combined_df['aum'].sum()
        summary_text.append(f"Total AUM Across All Funds: ${total_aum:,.0f}M")

    # Add Net Return Summary
    if "net_return_usd" in combined_df.columns and not combined_df["net_return_usd"].isnull().all():
        summary_text.append("")
        summary_text.append("Net Return Summary (USD Millions):")
        summary_text.append("")

        # Net Return By Strategy
        net_return_by_strategy = combined_df.groupby("strategy")["net_return_usd"].sum().sort_values(ascending=False)
        summary_text.append("By Strategy:")
        for strategy, net_ret in net_return_by_strategy.items():
            summary_text.append(f"- {strategy}: ${net_ret:,.2f}M")

        summary_text.append("")

        # Net Return By Fund
        net_return_by_fund = combined_df.groupby("fund_name")["net_return_usd"].sum().sort_values(ascending=False)
        summary_text.append("By Fund:")
        for fund, net_ret in net_return_by_fund.items():
            summary_text.append(f"- {fund}: ${net_ret:,.2f}M")


    # Add text to the figure
    y_pos = 0.9
    for line in summary_text:
        # Decrease font size slightly and adjust y_pos more if there are many lines
        if len(summary_text) > 15: # Arbitrary threshold for adjusting layout
            font_size = 10
            line_spacing = 0.04
        else:
            font_size = 12
            line_spacing = 0.05
        plt.text(0.05, y_pos, line, fontsize=font_size, ha='left', va='top', wrap=True)
        y_pos -= line_spacing

    plt.tight_layout()
    pdf.savefig(fig_text)
    plt.close(fig_text)

    # Close the PDF object
    pdf.close()
    print(f"\nSummary PDF report with multi-page charts and net return summary saved to: {summary_path}")

    # Export to Excel and Download Files
    combined_path = "combined_fund_report.xlsx"
    combined_df.to_excel(combined_path, index=False, sheet_name="FundData")
    print(f"Combined Excel report saved to: {combined_path}")

    print("Downloading generated reports...")
    files.download(combined_path)
    files.download(summary_path)
else:
    print("\n--- No valid data could be extracted from the uploaded files. ---")