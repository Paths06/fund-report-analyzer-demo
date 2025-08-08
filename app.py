#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import pandas as pd
import PyPDF2
import matplotlib.pyplot as plt
import json
import re
from io import BytesIO
from typing import Dict, List, Any, Optional
import base64
import matplotlib.backends.backend_pdf
from datetime import datetime, timedelta
import google.generativeai as genai
from google.generativeai import caching
import time
import hashlib

# Page configuration
st.set_page_config(
    page_title="AI-Powered Fund Report Analysis Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Fund Report Analysis Dashboard")
st.write("Upload your fund documents in any format. Our AI will intelligently extract and analyze fund performance data.")

# Initialize Gemini API
@st.cache_resource
def initialize_gemini():
    """Initialize Gemini API with API key from Streamlit secrets or environment"""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("⚠️ GEMINI_API_KEY not found. Please add it to your Streamlit secrets or environment variables.")
            st.stop()
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        st.stop()

# Context caching management
class ContextCache:
    """Manages context caching for Gemini API to optimize token usage"""
    
    def __init__(self):
        self.cache_store = {}
        self.cache_expiry = timedelta(hours=1)  # Cache expires after 1 hour
    
    def get_cache_key(self, content: str) -> str:
        """Generate a unique cache key based on content hash"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_context(self, content: str):
        """Retrieve cached context if available and not expired"""
        cache_key = self.get_cache_key(content)
        if cache_key in self.cache_store:
            cached_data, timestamp = self.cache_store[cache_key]
            if datetime.now() - timestamp < self.cache_expiry:
                return cached_data
            else:
                # Remove expired cache
                del self.cache_store[cache_key]
        return None
    
    def set_cached_context(self, content: str, context_data: Any):
        """Cache the context data"""
        cache_key = self.get_cache_key(content)
        self.cache_store[cache_key] = (context_data, datetime.now())
    
    def create_system_context(self) -> str:
        """Create the system context for fund analysis"""
        return """
        You are a senior financial analyst specializing in hedge fund and investment fund analysis. 
        Your task is to extract structured fund performance data from various document formats.
        
        EXTRACTION REQUIREMENTS:
        1. Fund Name: Complete fund name (clean and standardized)
        2. Return/Performance: Weekly, monthly, or period returns (convert to decimal, e.g., 5% = 0.05)
        3. AUM (Assets Under Management): In millions USD (convert B to thousands of millions)
        4. Strategy: Investment strategy category (standardized names)
        5. Additional Metrics: Any other relevant performance metrics
        
        STANDARDIZATION RULES:
        - Strategy names: Use standard categories like "Long/Short Equity", "Global Macro", "Event-Driven", etc.
        - AUM: Always in millions USD
        - Returns: Always as decimals (5% = 0.05)
        - Fund names: Clean, remove special characters and formatting artifacts
        
        OUTPUT FORMAT: Return data as a JSON array of objects with consistent field names:
        [
            {
                "fund_name": "Clean Fund Name",
                "return": 0.025,
                "aum": 150.5,
                "strategy": "Long/Short Equity",
                "additional_metrics": {}
            }
        ]
        
        QUALITY STANDARDS:
        - Extract ALL funds mentioned in the document
        - Ensure data consistency and accuracy
        - Handle missing data gracefully (use null for missing values)
        - Identify and skip header/footer information
        - Apply senior analyst judgment for ambiguous cases
        """

# Initialize cache
@st.cache_resource
def get_context_cache():
    return ContextCache()

# File processing functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF files"""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        st.warning(f"PDF extraction failed: {e}")
        return ""

def extract_text_from_excel(file_content: bytes, filename: str) -> str:
    """Convert Excel data to text representation"""
    try:
        # Try reading all sheets
        excel_data = pd.read_excel(BytesIO(file_content), sheet_name=None)
        text_content = []
        
        for sheet_name, df in excel_data.items():
            text_content.append(f"=== Sheet: {sheet_name} ===")
            text_content.append(df.to_string(index=False))
            text_content.append("\n")
        
        return "\n".join(text_content)
    except Exception as e:
        st.warning(f"Excel extraction failed: {e}")
        return ""

def extract_text_from_csv(file_content: bytes) -> str:
    """Extract text from CSV files"""
    try:
        df = pd.read_csv(BytesIO(file_content))
        return df.to_string(index=False)
    except Exception as e:
        st.warning(f"CSV extraction failed: {e}")
        return ""

def extract_text_content(file_content: bytes, filename: str) -> str:
    """Extract text content from various file formats"""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif filename_lower.endswith(('.xlsx', '.xls')):
        return extract_text_from_excel(file_content, filename)
    elif filename_lower.endswith('.csv'):
        return extract_text_from_csv(file_content)
    elif filename_lower.endswith('.txt'):
        return file_content.decode('utf-8', errors='ignore')
    else:
        # Try to decode as text for unknown formats
        try:
            return file_content.decode('utf-8', errors='ignore')
        except:
            st.warning(f"Unsupported file format: {filename}")
            return ""

def extract_fund_data_with_gemini(model, text_content: str, filename: str, cache: ContextCache) -> pd.DataFrame:
    """Extract fund data using Gemini AI with context caching"""
    
    # Check cache first
    cached_result = cache.get_cached_context(text_content)
    if cached_result is not None:
        st.info(f"Using cached analysis for {filename}")
        return cached_result
    
    try:
        system_context = cache.create_system_context()
        
        prompt = f"""
        {system_context}
        
        DOCUMENT TO ANALYZE: {filename}
        
        CONTENT:
        {text_content[:8000]}  # Limit content to avoid token limits
        
        Please extract all fund performance data from this document and return as valid JSON array.
        Focus on accuracy and completeness. Apply senior analyst expertise to interpret the data correctly.
        """
        
        with st.spinner(f"AI analyzing {filename}..."):
            response = model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean up response to extract JSON
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end]
            elif '[' in response_text and ']' in response_text:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            # Parse JSON
            fund_data = json.loads(json_text)
            
            # Convert to DataFrame
            df = pd.DataFrame(fund_data)
            
            # Cache the result
            cache.set_cached_context(text_content, df)
            
            st.success(f"✅ Extracted {len(df)} funds from {filename}")
            return df
            
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response for {filename}: {e}")
        st.write("Raw AI Response:", response.text[:500])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"AI extraction failed for {filename}: {e}")
        return pd.DataFrame()

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the extracted DataFrame"""
    if df.empty:
        return df
    
    # Ensure required columns exist
    required_columns = ['fund_name', 'return', 'strategy']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Add AUM column if missing
    if 'aum' not in df.columns:
        df['aum'] = None
    
    # Data cleaning and standardization
    df['fund_name'] = df['fund_name'].astype(str).str.strip()
    
    # Convert returns to numeric
    if 'return' in df.columns:
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
    
    # Convert AUM to numeric
    if 'aum' in df.columns:
        df['aum'] = pd.to_numeric(df['aum'], errors='coerce')
    
    # Clean strategy names
    if 'strategy' in df.columns:
        df['strategy'] = df['strategy'].astype(str).str.strip()
    
    # Calculate net return in USD if both AUM and return are available
    if 'aum' in df.columns and 'return' in df.columns:
        df['net_return_usd'] = df['return'] * df['aum']
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['fund_name'])
    
    return df

def create_visualizations(df: pd.DataFrame):
    """Create comprehensive visualizations"""
    if df.empty:
        st.warning("No data available for visualization")
        return
    
    st.subheader("📊 Performance Analytics")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["Returns Analysis", "AUM Analysis", "Strategy Performance", "Risk Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.write("**Top Performing Funds**")
                top_funds = df.nlargest(10, 'return')[['fund_name', 'return', 'strategy']]
                top_funds['return'] = top_funds['return'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                st.dataframe(top_funds, use_container_width=True)
        
        with col2:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.write("**Return Distribution**")
                fig, ax = plt.subplots(figsize=(8, 6))
                returns_clean = df['return'].dropna()
                ax.hist(returns_clean, bins=20, alpha=0.7, color='skyblue')
                ax.set_xlabel('Return')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Fund Returns')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        if 'aum' in df.columns and not df['aum'].isnull().all():
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Largest Funds by AUM**")
                large_funds = df.nlargest(10, 'aum')[['fund_name', 'aum', 'strategy']]
                large_funds['aum'] = large_funds['aum'].apply(lambda x: f"${x:,.1f}M" if pd.notna(x) else "N/A")
                st.dataframe(large_funds, use_container_width=True)
            
            with col2:
                st.write("**AUM by Strategy**")
                aum_by_strategy = df.groupby('strategy')['aum'].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(range(len(aum_by_strategy)), aum_by_strategy.values, color='orange')
                ax.set_xticks(range(len(aum_by_strategy)))
                ax.set_xticklabels(aum_by_strategy.index, rotation=45, ha='right')
                ax.set_ylabel('Total AUM (M USD)')
                ax.set_title('Total AUM by Strategy')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        if 'strategy' in df.columns and 'return' in df.columns:
            strategy_stats = df.groupby('strategy').agg({
                'return': ['mean', 'std', 'count'],
                'aum': 'sum'
            }).round(4)
            
            strategy_stats.columns = ['Avg Return', 'Return Std', 'Fund Count', 'Total AUM']
            strategy_stats['Avg Return'] = strategy_stats['Avg Return'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            strategy_stats['Return Std'] = strategy_stats['Return Std'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            strategy_stats['Total AUM'] = strategy_stats['Total AUM'].apply(lambda x: f"${x:,.1f}M" if pd.notna(x) else "N/A")
            
            st.write("**Strategy Performance Summary**")
            st.dataframe(strategy_stats, use_container_width=True)
    
    with tab4:
        if 'return' in df.columns and not df['return'].isnull().all():
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk-Return Scatter
                if 'aum' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(df['return'], df['aum'], alpha=0.6, s=50)
                    ax.set_xlabel('Return')
                    ax.set_ylabel('AUM (M USD)')
                    ax.set_title('Risk-Return Profile')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                # Performance metrics
                returns_clean = df['return'].dropna()
                if len(returns_clean) > 0:
                    metrics = {
                        'Average Return': f"{returns_clean.mean():.2%}",
                        'Median Return': f"{returns_clean.median():.2%}",
                        'Standard Deviation': f"{returns_clean.std():.2%}",
                        'Best Performer': f"{returns_clean.max():.2%}",
                        'Worst Performer': f"{returns_clean.min():.2%}"
                    }
                    
                    st.write("**Portfolio Metrics**")
                    for metric, value in metrics.items():
                        st.metric(metric, value)

def generate_executive_summary_pdf(df: pd.DataFrame) -> bytes:
    """Generate a professional one-page executive summary PDF"""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.patches as mpatches
    from datetime import datetime
    
    # Create PDF
    pdf_buffer = BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Create figure with custom layout
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
        fig.patch.set_facecolor('white')
        
        # Title and header
        fig.suptitle('FUND PERFORMANCE EXECUTIVE SUMMARY', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Date and summary stats
        report_date = datetime.now().strftime('%B %d, %Y')
        fig.text(0.1, 0.91, f'Report Date: {report_date}', fontsize=10)
        fig.text(0.6, 0.91, f'Funds Analyzed: {len(df)} | Strategies: {df["strategy"].nunique() if "strategy" in df.columns else "N/A"}', fontsize=10)
        
        # Key Highlights Section
        ax1 = fig.add_subplot(4, 2, (1, 2))
        ax1.axis('off')
        ax1.text(0.02, 0.9, 'KEY HIGHLIGHTS', fontsize=14, fontweight='bold', 
                transform=ax1.transAxes)
        
        highlights = []
        
        if 'return' in df.columns and not df['return'].isnull().all():
            avg_return = df['return'].mean()
            best_fund = df.loc[df['return'].idxmax(), 'fund_name'] if not df['return'].isnull().all() else "N/A"
            best_return = df['return'].max()
            worst_return = df['return'].min()
            
            highlights.extend([
                f"• Portfolio Average Return: {avg_return:.2%}",
                f"• Best Performer: {best_fund} ({best_return:.2%})",
                f"• Return Range: {worst_return:.2%} to {best_return:.2%}",
            ])
        
        if 'aum' in df.columns and not df['aum'].isnull().all():
            total_aum = df['aum'].sum()
            largest_fund = df.loc[df['aum'].idxmax(), 'fund_name'] if not df['aum'].isnull().all() else "N/A"
            highlights.extend([
                f"• Total AUM: ${total_aum:,.0f}M",
                f"• Largest Fund: {largest_fund}",
            ])
        
        if 'strategy' in df.columns:
            top_strategy = df.groupby('strategy')['return'].mean().idxmax() if 'return' in df.columns else "N/A"
            strategy_count = df.groupby('strategy').size().max()
            highlights.append(f"• Top Strategy: {top_strategy}")
            highlights.append(f"• Most Common Strategy: {strategy_count} funds")
        
        # Display highlights
        for i, highlight in enumerate(highlights):
            ax1.text(0.02, 0.75 - i*0.1, highlight, fontsize=11, 
                    transform=ax1.transAxes)
        
        # Top Performers Table
        ax2 = fig.add_subplot(4, 2, (3, 4))
        ax2.axis('off')
        ax2.text(0.02, 0.9, 'TOP PERFORMERS', fontsize=14, fontweight='bold', 
                transform=ax2.transAxes)
        
        if 'return' in df.columns and not df['return'].isnull().all():
            top_performers = df.nlargest(5, 'return')[['fund_name', 'return', 'strategy']].copy()
            top_performers['return'] = top_performers['return'].apply(lambda x: f"{x:.2%}")
            
            # Create table
            table_data = []
            for _, row in top_performers.iterrows():
                fund_name = row['fund_name'][:25] + "..." if len(row['fund_name']) > 25 else row['fund_name']
                strategy = row['strategy'][:15] + "..." if len(str(row['strategy'])) > 15 else row['strategy']
                table_data.append([fund_name, row['return'], strategy])
            
            table = ax2.table(cellText=table_data,
                            colLabels=['Fund Name', 'Return', 'Strategy'],
                            cellLoc='left',
                            loc='center',
                            bbox=[0.02, 0.1, 0.96, 0.7])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
        
        # Strategy Performance Chart
        ax3 = fig.add_subplot(4, 2, 5)
        if 'strategy' in df.columns and 'return' in df.columns:
            strategy_returns = df.groupby('strategy')['return'].mean().sort_values(ascending=True)
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(strategy_returns)))
            
            bars = ax3.barh(range(len(strategy_returns)), strategy_returns.values, color=colors)
            ax3.set_yticks(range(len(strategy_returns)))
            ax3.set_yticklabels([s[:12] + "..." if len(s) > 12 else s for s in strategy_returns.index], fontsize=9)
            ax3.set_xlabel('Average Return', fontsize=10)
            ax3.set_title('Strategy Performance', fontsize=12, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(strategy_returns.values):
                ax3.text(v + 0.001, i, f'{v:.1%}', va='center', fontsize=8)
        
        # AUM Distribution Chart
        ax4 = fig.add_subplot(4, 2, 6)
        if 'aum' in df.columns and not df['aum'].isnull().all():
            aum_by_strategy = df.groupby('strategy')['aum'].sum().sort_values(ascending=True)
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(aum_by_strategy)))
            
            bars = ax4.barh(range(len(aum_by_strategy)), aum_by_strategy.values, color=colors)
            ax4.set_yticks(range(len(aum_by_strategy)))
            ax4.set_yticklabels([s[:12] + "..." if len(s) > 12 else s for s in aum_by_strategy.index], fontsize=9)
            ax4.set_xlabel('Total AUM ($M)', fontsize=10)
            ax4.set_title('AUM by Strategy', fontsize=12, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(aum_by_strategy.values):
                ax4.text(v + max(aum_by_strategy.values)*0.01, i, f'${v:.0f}M', va='center', fontsize=8)
        
        # Risk Analysis Section
        ax5 = fig.add_subplot(4, 2, (7, 8))
        ax5.axis('off')
        ax5.text(0.02, 0.9, 'RISK ANALYSIS & INSIGHTS', fontsize=14, fontweight='bold', 
                transform=ax5.transAxes)
        
        risk_insights = []
        
        if 'return' in df.columns and not df['return'].isnull().all():
            returns_std = df['return'].std()
            positive_returns = (df['return'] > 0).sum()
            total_funds = len(df[df['return'].notna()])
            
            risk_insights.extend([
                f"• Return Volatility: {returns_std:.2%}",
                f"• Positive Returns: {positive_returns}/{total_funds} funds ({positive_returns/total_funds:.1%})",
            ])
            
            # Sharpe-like ratio (assuming risk-free rate of 2%)
            excess_return = df['return'].mean() - 0.02
            if returns_std > 0:
                risk_adj_return = excess_return / returns_std
                risk_insights.append(f"• Risk-Adjusted Return Ratio: {risk_adj_return:.2f}")
        
        if 'strategy' in df.columns:
            strategy_consistency = df.groupby('strategy')['return'].std().min()
            most_consistent = df.groupby('strategy')['return'].std().idxmin()
            risk_insights.append(f"• Most Consistent Strategy: {most_consistent}")
        
        # Display risk insights
        for i, insight in enumerate(risk_insights):
            ax5.text(0.02, 0.7 - i*0.12, insight, fontsize=11, 
                    transform=ax5.transAxes)
        
        # Footer
        fig.text(0.1, 0.02, 'Generated by AI-Powered Fund Analysis Dashboard', 
                fontsize=8, style='italic', alpha=0.7)
        fig.text(0.7, 0.02, f'Page 1 of 1', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    return pdf_buffer.getvalue()

def create_download_links(df: pd.DataFrame):
    """Create download links for reports including executive summary PDF"""
    st.subheader("⬇️ Download Reports")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Executive Summary PDF
        if st.button("📄 Generate Executive Summary"):
            with st.spinner("Creating executive summary..."):
                try:
                    pdf_bytes = generate_executive_summary_pdf(df)
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="executive_summary.pdf">📄 Download Executive Summary PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Executive summary generated!")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
    
    with col2:
        # Excel download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Fund Analysis')
            
            # Add summary sheet
            if 'strategy' in df.columns:
                summary = df.groupby('strategy').agg({
                    'return': ['mean', 'std', 'count'],
                    'aum': 'sum'
                }).round(4)
                summary.to_excel(writer, sheet_name='Strategy Summary')
        
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="fund_analysis.xlsx">📊 Download Excel Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        # CSV download
        csv_data = df.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="fund_data.csv">📋 Download CSV Data</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col4:
        # JSON download
        json_data = df.to_json(orient='records', indent=2)
        b64 = base64.b64encode(json_data.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="fund_data.json">🔧 Download JSON Data</a>'
        st.markdown(href, unsafe_allow_html=True)

# Main application
def main():
    # Initialize Gemini model and cache
    model = initialize_gemini()
    cache = get_context_cache()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API usage info
        st.info("💡 **Smart Context Caching**: The app caches AI analysis to minimize token usage and improve response times.")
        
        # Cache management
        if st.button("🗑️ Clear Analysis Cache"):
            cache.cache_store.clear()
            st.success("Cache cleared!")
        
        # Display cache stats
        st.write(f"**Cached Files:** {len(cache.cache_store)}")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Fund Documents", 
        accept_multiple_files=True,
        type=['pdf', 'xlsx', 'xls', 'csv', 'txt'],
        help="Supported formats: PDF, Excel, CSV, Text files"
    )
    
    if uploaded_files:
        st.write(f"📁 Processing {len(uploaded_files)} files...")
        
        # Process files with progress bar
        all_dataframes = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            file_content = uploaded_file.read()
            filename = uploaded_file.name
            
            # Extract text content
            text_content = extract_text_content(file_content, filename)
            
            if text_content.strip():
                # Use Gemini to extract fund data
                df = extract_fund_data_with_gemini(model, text_content, filename, cache)
                
                if not df.empty:
                    df = standardize_dataframe(df)
                    df['source_file'] = filename  # Track source file
                    all_dataframes.append(df)
                else:
                    st.warning(f"⚠️ No fund data extracted from {filename}")
            else:
                st.error(f"❌ Could not extract text from {filename}")
        
        progress_bar.empty()
        
        # Combine and analyze data
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Display results
            st.success(f"🎉 Successfully processed {len(all_dataframes)} files")
            st.subheader("📋 Extracted Fund Data")
            st.dataframe(combined_df, use_container_width=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Funds", len(combined_df))
            
            with col2:
                if 'strategy' in combined_df.columns:
                    st.metric("Strategies", combined_df['strategy'].nunique())
            
            with col3:
                if 'aum' in combined_df.columns and not combined_df['aum'].isnull().all():
                    total_aum = combined_df['aum'].sum()
                    st.metric("Total AUM", f"${total_aum:,.0f}M")
            
            with col4:
                if 'return' in combined_df.columns and not combined_df['return'].isnull().all():
                    avg_return = combined_df['return'].mean()
                    st.metric("Avg Return", f"{avg_return:.2%}")
            
            # Create visualizations
            create_visualizations(combined_df)
            
            # Download options
            create_download_links(combined_df)
            
        else:
            st.error("❌ No valid fund data could be extracted from any of the uploaded files.")
            st.info("💡 **Tips for better results:**\n- Ensure files contain fund performance data\n- Check that text is not image-based in PDFs\n- Verify data format is readable")
    
    else:
        # Welcome message and instructions
        st.info("""
        🚀 **Get Started:**
        
        1. **Upload Files**: Support for PDF, Excel, CSV, and text files
        2. **AI Analysis**: Our AI will intelligently extract fund data regardless of format
        3. **Smart Insights**: Get professional-grade analysis and visualizations
        4. **Export Results**: Download comprehensive reports in multiple formats
        
        **What the AI can extract:**
        - Fund names and performance returns
        - Assets Under Management (AUM)
        - Investment strategies
        - Additional performance metrics
        
        **Built for analysts by analysts** - leveraging advanced AI to handle diverse document formats and complex data structures.
        """)

if __name__ == "__main__":
    main()
