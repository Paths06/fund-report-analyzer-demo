#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import pandas as pd
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import base64
import numpy as np
from io import BytesIO
from typing import Dict, List, Any, Optional
import matplotlib.backends.backend_pdf
from datetime import datetime, timedelta
import google.generativeai as genai
from google.generativeai import caching
import time
import hashlib

# Dark theme configuration and styling
st.set_page_config(
    page_title="Fund Report Analysis Dashboard",
    # page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app background and text colors */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1d29 50%, #0f1419 100%);
        color: #e8eaed;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e2329;
        border-right: 1px solid #333740;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(30, 35, 41, 0.6);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem;
    }
    
    /* Headers and titles */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #64b5f6, #42a5f5, #2196f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #263238 0%, #37474f 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #404040;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #64b5f6;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #b0bec5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Cards and containers */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2c3e50;
        border-radius: 8px;
        padding: 0.2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #b0bec5;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #42a5f5 !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #42a5f5 0%, #2196f3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(66, 165, 245, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(66, 165, 245, 0.4);
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #263238;
        border: 2px dashed #42a5f5;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stFileUploader > div {
        background-color: transparent;
    }
    
    /* DataFrames */
    .stDataFrame {
        background-color: #1e2329;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #333740;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background-color: #263238;
    }
    
    .stProgress .st-bp {
        background-color: #42a5f5;
    }
    
    /* Alerts and info boxes */
    .stAlert {
        background-color: rgba(66, 165, 245, 0.1);
        border: 1px solid #42a5f5;
        border-radius: 8px;
        color: #e8eaed;
    }
    
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1);
        border: 1px solid #4caf50;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
    }
    
    .stError {
        background-color: rgba(244, 67, 54, 0.1);
        border: 1px solid #f44336;
    }
    
    /* Sidebar elements */
    .css-1lcbmhc .css-1outpf7 {
        background-color: #2c3e50;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #404040;
    }
    
    /* Text and paragraph styling */
    p, .stMarkdown {
        color: #b0bec5 !important;
        line-height: 1.6;
    }
    
    /* Links */
    a {
        color: #64b5f6 !important;
        text-decoration: none;
    }
    
    a:hover {
        color: #42a5f5 !important;
        text-decoration: underline;
    }
    
    /* Professional gradient backgrounds for sections */
    .analysis-section {
        background: linear-gradient(135deg, rgba(66, 165, 245, 0.05) 0%, rgba(33, 150, 243, 0.05) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(66, 165, 245, 0.2);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e2329;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #42a5f5;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced title with icon and subtitle
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1>🤖 AI-Powered Fund Report Analysis Dashboard</h1>
    <p style="font-size: 1.1rem; color: #90a4ae; margin-top: -1rem;">
        fund analysis Dashboard • Extract insights from any document format
    </p>
</div>
""", unsafe_allow_html=True)

# Dark theme matplotlib setup
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#1e2329',
    'axes.facecolor': '#263238',
    'axes.edgecolor': '#404040',
    'axes.labelcolor': '#e8eaed',
    'text.color': '#e8eaed',
    'xtick.color': '#b0bec5',
    'ytick.color': '#b0bec5',
    'grid.color': '#404040',
    'font.family': 'Inter',
    'font.size': 10
})

# Set seaborn dark theme
sns.set_theme(style="darkgrid", palette="viridis")
sns.set_context("notebook", font_scale=1.1)

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

@st.cache_resource
def get_context_cache():
    """Get or create a context cache instance"""
    return ContextCache()
    
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
        """Create the original system context for fund analysis"""
        return """
        You are a senior financial analyst specializing in hedge fund and investment fund analysis. 
        Your task is to extract structured fund performance data from various document formats.
        
        EXTRACTION REQUIREMENTS:
        1. Fund Name: Complete fund name (clean and standardized)
        2. Return/Performance: Weekly, monthly, or period returns (convert to decimal, e.g., 5% = 0.05)
        3. AUM (Assets Under Management): In millions USD (convert B to thousands of millions)
        4. Strategy: Investment strategy category (standardized names) - NEVER leave this as null
        5. Additional Metrics: Any other relevant performance metrics
        
        STANDARDIZATION RULES:
        - Strategy names: Use standard categories like "Long/Short Equity", "Global Macro", "Event-Driven", "Quantitative", "Fixed Income Arbitrage", "Credit", "Multi-Strategy"
        - If strategy is unclear, infer from fund name or context
        - Common mappings: "L/S Equity" -> "Long/Short Equity", "L/S Eq" -> "Long/Short Equity", "Quant" -> "Quantitative", "Fixed Income Arb" -> "Fixed Income Arbitrage"
        - AUM: Always in millions USD (convert B to 1000, so 1.1B = 1100)
        - Returns: Always as decimals (5% = 0.05, 1.45% = 0.0145)
        - Fund names: Clean, remove special characters and formatting artifacts
        
        CRITICAL: Every fund MUST have a strategy. If not explicitly stated, infer from:
        - Fund name (e.g., "Credit" in name -> "Credit" strategy)
        - Context clues in the document
        - Default to "Multi-Strategy" only if absolutely no indication
        
        OUTPUT FORMAT: Return data as a JSON array of objects with consistent field names:
        [
            {
                "fund_name": "Clean Fund Name",
                "return": 0.0145,
                "aum": 520.0,
                "strategy": "Quantitative",
                "additional_metrics": {}
            }
        ]
        
        QUALITY STANDARDS:
        - Extract ALL funds mentioned in the document
        - Ensure data consistency and accuracy
        - Handle missing data gracefully (use null for missing AUM/return, but NEVER for strategy)
        - Identify and skip header/footer information
        - Apply senior analyst judgment for ambiguous cases
        - Double-check strategy assignments - no fund should have null/empty strategy
        """
    
    def create_robust_system_context(self) -> str:
        """Create enhanced system context for handling any report format"""
        return """
        You are a senior financial analyst with 20+ years of experience analyzing hedge fund and investment reports. 
        Your expertise includes reading ANY type of fund document regardless of format, structure, or terminology.
        
        CORE MISSION: Extract fund performance data from ANY document format, even if poorly structured.
        
        EXTRACTION REQUIREMENTS (Priority Order):
        1. Fund Name: ANY fund identifier (clean and standardize)
        2. Performance/Return: ANY performance metric (%, basis points, absolute numbers)
        3. Assets/Capital: ANY size metric (AUM, NAV, Assets, Capital, etc.)
        4. Strategy/Style: ANY strategy indication (infer if not explicit)
        5. Additional Data: ANY other relevant metrics
        
        TERMINOLOGY FLEXIBILITY - Recognize ALL variations:
        
        FUND NAMES:
        - "Fund", "Strategy", "Product", "Vehicle", "Account", "Portfolio"
        - "LP", "Ltd", "Fund I/II/III", "Class A/B/C"
        - Any proper nouns followed by investment terms
        
        PERFORMANCE TERMS:
        - "Return", "Performance", "Gain/Loss", "P&L", "Net Return", "Gross Return"
        - "MTD", "QTD", "YTD", "ITD", "Since Inception"
        - "Weekly", "Monthly", "Quarterly", "Annual"
        - Numbers with %, bp, $ symbols
        
        AUM/SIZE TERMS:
        - "AUM", "Assets Under Management", "Net Assets", "NAV", "Capital"
        - "Fund Size", "Total Assets", "Gross Assets", "Market Value"
        - "Commitments", "Subscriptions", "Capital Base"
        - ANY number with M, MM, B, K, Million, Billion suffixes
        
        STRATEGY TERMS:
        - Explicit: "Strategy", "Style", "Approach", "Focus", "Sector"
        - Implicit: Look for keywords in fund names, descriptions
        - Geographic: "US", "Europe", "Asia", "Global", "Emerging Markets"
        - Asset Class: "Equity", "Credit", "Fixed Income", "Commodities", "Real Estate"
        - Style: "Long/Short", "Market Neutral", "Distressed", "Growth", "Value"
        
        INTELLIGENT INFERENCE RULES:
        
        1. FUND NAME INFERENCE:
        - If no explicit fund name, use document title or header
        - Look for recurring names with financial terms
        - Extract from table headers or section titles
        
        2. PERFORMANCE INFERENCE:
        - Convert ANY percentage format to decimal (5% = 0.05, 500bp = 0.05)
        - Handle negative returns correctly
        - If only absolute $ amounts, calculate percentage if possible
        - Default to most recent period if multiple timeframes
        
        3. AUM INFERENCE:
        - Convert ALL units to millions USD consistently
        - Handle: K=thousands, M=millions, MM=millions, B=billions
        - If only NAV per share, multiply by shares outstanding if available
        - If multiple size metrics, prioritize AUM > Net Assets > Total Assets
        
        4. STRATEGY INFERENCE FROM FUND NAMES:
        - "Credit", "Debt" → "Credit"
        - "Equity", "Stock", "Long/Short" → "Long/Short Equity"
        - "Macro", "Currency", "FX" → "Global Macro"
        - "Merger", "Event", "Special Situations" → "Event-Driven"
        - "Multi", "Diversified", "Flexible" → "Multi-Strategy"
        - "Quant", "Systematic", "Algorithm" → "Quantitative"
        - "Real Estate", "REIT" → "Real Estate"
        - "Commodity", "Energy", "Agriculture" → "Commodities"
        - Geographic terms → add to strategy (e.g., "European Long/Short Equity")
        
        ERROR HANDLING & FALLBACKS:
        - If data is unclear, make best professional judgment
        - If AUM missing, use null but extract everything else
        - If strategy unclear, infer from fund name or use "Multi-Strategy"
        - If return format is unclear, document what was found
        - Never skip a fund due to missing data - extract what you can
        
        DOCUMENT STRUCTURE HANDLING:
        - Handle tables with merged cells, multiple headers
        - Extract from footnotes, appendices, summary sections
        - Parse narrative text for embedded data
        - Handle multiple funds per document vs single fund reports
        - Extract from charts/graphs if data is embedded in text
        
        QUALITY ASSURANCE:
        - Sanity check: Returns typically between -50% to +200%
        - Sanity check: AUM typically between $1M to $100B
        - Flag unusual values but don't discard
        - Cross-reference fund names for consistency
        - Ensure strategy assignments make sense
        
        OUTPUT FORMAT (ALWAYS JSON):
        [
            {
                "fund_name": "Clean Fund Name",
                "return": 0.0145,  // Always decimal format
                "aum": 520.0,      // Always in millions USD
                "strategy": "Quantitative",  // Never null/empty
                "period": "Monthly",  // If determinable
                "confidence": "High",  // High/Medium/Low based on data clarity
                "additional_metrics": {
                    "data_source": "Page 1 table",
                    "original_return_format": "1.45%",
                    "original_aum_format": "$520M"
                }
            }
        ]
        
        CRITICAL SUCCESS FACTORS:
        1. Extract SOMETHING from every document, even if partial
        2. Never leave strategy field empty - always infer
        3. Be aggressive in finding data - check headers, footers, appendices
        4. Apply 20+ years of fund industry knowledge
        5. When in doubt, make the most reasonable professional assumption
        
        Remember: You're dealing with real investment documents that may be:
        - Poorly formatted presentations
        - Scanned PDFs with OCR errors
        - Excel files with complex structures
        - Email updates with embedded data
        - Regulatory filings with specific formats
        - Marketing materials with limited data
        
        Your job is to be the expert human analyst who can extract meaningful data from ANY of these formats.
        """
# Enhanced file processing functions with dark theme status updates

def extract_text_content_enhanced(file_content: bytes, filename: str) -> str:
        """Enhanced text extraction with multiple fallback methods"""
        filename_lower = filename.lower()
        
        text_content = ""
        
        # Enhanced status display with icons
        with st.status(f"📄 Processing {filename}...", expanded=False) as status:
            if filename_lower.endswith('.pdf'):
                # status.write("🔍 Extracting PDF content...")
                text_content = extract_pdf_with_fallbacks(file_content)
            elif filename_lower.endswith(('.xlsx', '.xls')):
                # status.write("📊 Parsing Excel spreadsheet...")
                text_content = extract_excel_enhanced(file_content, filename)
            elif filename_lower.endswith('.csv'):
                # status.write("📈 Reading CSV data...")
                text_content = extract_csv_enhanced(file_content)
            elif filename_lower.endswith('.txt'):
                # status.write("📝 Reading text file...")
                text_content = file_content.decode('utf-8', errors='ignore')
            else:
                # status.write("🔧 Attempting generic text extraction...")
                try:
                    text_content = file_content.decode('utf-8', errors='ignore')
                except:
                    st.warning(f"⚠️ Unsupported file format: {filename}")
                    return ""
            
            if len(text_content.strip()) < 50:
                status.update(label=f"⚠️ Limited content extracted from {filename}", state="warning")
                st.warning(f"Limited text extracted from {filename}. Trying alternate methods...")
            else:
                status.update(label=f"✅ Successfully processed {filename}", state="complete")
        
        return text_content

def extract_pdf_with_fallbacks(file_content: bytes) -> str:
        """Enhanced PDF extraction with multiple methods"""
        text = ""
        
        try:
            reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            if len(text.strip()) < 50:
                st.info("💡 PyPDF2 extraction yielded minimal text. Document may be image-based.")
                
        except Exception as e:
            st.warning(f"⚠️ PDF extraction failed with PyPDF2: {e}")
        
        return text

def extract_excel_enhanced(file_content: bytes, filename: str) -> str:
        """Enhanced Excel extraction handling complex structures"""
        try:
            excel_data = pd.read_excel(BytesIO(file_content), sheet_name=None, header=None)
            text_content = []
            
            for sheet_name, df in excel_data.items():
                text_content.append(f"=== Sheet: {sheet_name} ===")
                
                for i, row in df.iterrows():
                    row_text = " | ".join([str(cell) for cell in row.dropna()])
                    if len(row_text.strip()) > 0:
                        text_content.append(row_text)
                
                text_content.append("\n")
            
            return "\n".join(text_content)
        except Exception as e:
            st.warning(f"⚠️ Enhanced Excel extraction failed: {e}")
            try:
                df = pd.read_excel(BytesIO(file_content))
                return df.to_string(index=False)
            except:
                return ""

def extract_csv_enhanced(file_content: bytes) -> str:
        """Enhanced CSV extraction with encoding detection"""
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
                try:
                    df = pd.read_csv(BytesIO(file_content), encoding=encoding)
                    return df.to_string(index=False)
                except:
                    continue
            
            df = pd.read_csv(BytesIO(file_content), encoding='utf-8', errors='ignore')
            return df.to_string(index=False)
            
        except Exception as e:
            st.warning(f"⚠️ CSV extraction failed: {e}")
            return ""

def extract_fund_data_with_gemini_enhanced(model, text_content: str, filename: str, cache: ContextCache) -> pd.DataFrame:
        """Enhanced Gemini extraction with better error handling and validation"""
        
        # Check cache first with enhanced UI
        cached_result = cache.get_cached_context(text_content)
        if cached_result is not None:
            st.success(f"⚡ Using cached analysis for {filename}")
            return cached_result
        
        try:
            system_context = cache.create_robust_system_context()
            
            prompt = f"""
            {system_context}
            
            DOCUMENT TO ANALYZE: {filename}
            DOCUMENT TYPE: {filename.split('.')[-1].upper() if '.' in filename else 'Unknown'}
            
            CONTENT (First 12,000 characters):
            {text_content[:12000]}
            
            ANALYSIS INSTRUCTIONS:
            1. This is a real investment document that may have ANY format
            2. Apply your 20+ years of fund industry expertise
            3. Extract ALL possible fund data, even if structure is unclear
            4. Make professional judgments when data is ambiguous
            5. Never skip funds due to missing data - extract what you can
            6. If you see numbers that could be performance, include them
            7. If you see fund-like names, include them
            8. Use your experience to infer missing information
            
            Return comprehensive JSON with ALL funds found, including confidence levels.
            """
            
            # Enhanced spinner with more professional styling
            with st.spinner(f" analyzing {filename} with advanced intelligence..."):
                response = model.generate_content(prompt)
                
                response_text = response.text.strip()
                json_text = None
                
                # Method 1: Look for ```json blocks
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end]
                
                # Method 2: Look for [ ] array
                elif '[' in response_text and ']' in response_text:
                    json_start = response_text.find('[')
                    json_end = response_text.rfind(']') + 1
                    json_text = response_text[json_start:json_end]
                
                # Method 3: Try to clean up the response
                else:
                    cleaned = response_text.replace("Here's the extracted data:", "")
                    cleaned = cleaned.replace("Based on the document:", "")
                    json_pattern = r'\[.*?\]'
                    matches = re.findall(json_pattern, cleaned, re.DOTALL)
                    if matches:
                        json_text = matches[0]
                
                if json_text is None:
                    st.error(f"❌ Could not find valid JSON in response for {filename}")
                    with st.expander("🔍 Debug: Raw Response"):
                        st.code(response_text[:1000], language="text")
                    return pd.DataFrame()
                
                # Parse JSON with enhanced error handling
                try:
                    fund_data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    st.warning(f"⚠️ JSON parsing failed for {filename}. Attempting to fix...")
                    
                    # Try to fix common JSON issues
                    json_text = json_text.replace("'", '"')  # Replace single quotes
                    json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                    json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas
                    
                    # Remove JSON comments (// style comments)
                    json_text = re.sub(r'//.*?(?=\n|$)', '', json_text)
                    
                    # Remove /* */ style comments
                    json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
                    
                    try:
                        fund_data = json.loads(json_text)
                        st.success("✅ Successfully fixed JSON formatting issues")
                    except:
                        st.error(f"❌ Could not parse JSON for {filename}")
                        with st.expander("🔍 Debug: Cleaned JSON"):
                            st.code(json_text[:1000], language="json")
                        return pd.DataFrame()
                
                df = pd.DataFrame(fund_data)
                
                if len(df) > 0:
                    df = validate_and_clean_extracted_data(df, filename)
                
                # Cache the result
                cache.set_cached_context(text_content, df)
                
                # Enhanced success message with confidence breakdown
                confidence_info = ""
                if 'confidence' in df.columns:
                    high_conf = (df['confidence'] == 'High').sum()
                    med_conf = (df['confidence'] == 'Medium').sum()
                    low_conf = (df['confidence'] == 'Low').sum()
                    confidence_info = f" (📊 Confidence: {high_conf}H/{med_conf}M/{low_conf}L)"
                
                st.success(f"✅ Extracted {len(df)} funds from {filename}{confidence_info}")
                
                # Show extraction summary in expandable section
                if len(df) > 0:
                    with st.expander(f"📋 Extraction Summary for {filename}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Funds Extracted", len(df))
                        with col2:
                            if 'strategy' in df.columns:
                                st.metric("Strategies Found", df['strategy'].nunique())
                        with col3:
                            if 'aum' in df.columns and not df['aum'].isnull().all():
                                total_aum = df['aum'].sum()
                                st.metric("Total AUM", f"${total_aum:,.0f}M")
                
                return df
                
        except Exception as e:
            st.error(f"❌ Extraction failed for {filename}: {e}")
            
            # Enhanced error information with troubleshooting tips
            with st.expander("🔧 Troubleshooting Information"):
                st.write("**Possible causes:**")
                st.write("• Document is image-based and needs OCR")
                st.write("• Document has unusual structure")
                st.write("• API rate limits or connection issues")
                st.write("• Insufficient text content extracted")
                
                st.write("**Suggested solutions:**")
                st.write("• Try converting PDF to text format")
                st.write("• Ensure document contains actual fund data")
                st.write("• Check your Gemini API quota")
            
            return pd.DataFrame()

def validate_and_clean_extracted_data(df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Validate and clean extracted fund data with enhanced reporting"""
        
        if df.empty:
            return df
        
        original_count = len(df)
        
        # Ensure required columns exist
        required_columns = ['fund_name', 'return', 'strategy']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Add missing columns
        if 'aum' not in df.columns:
            df['aum'] = None
        if 'confidence' not in df.columns:
            df['confidence'] = 'Medium'
        
        # Clean fund names
        df['fund_name'] = df['fund_name'].astype(str).str.strip()
        df = df[df['fund_name'] != 'nan']
        df = df[df['fund_name'].str.len() > 2]
        
        # Clean and validate returns with enhanced reporting
        if 'return' in df.columns:
            df['return'] = pd.to_numeric(df['return'], errors='coerce')
            suspicious_returns = df[(df['return'].abs() > 2.0) & df['return'].notna()]
            if len(suspicious_returns) > 0:
                st.warning(f"⚠️ Found {len(suspicious_returns)} funds with returns > 200% in {filename}")
                with st.expander("🔍 View Suspicious Returns"):
                    st.dataframe(suspicious_returns[['fund_name', 'return']], use_container_width=True)
        
        # Clean and validate AUM with enhanced reporting
        if 'aum' in df.columns:
            df['aum'] = pd.to_numeric(df['aum'], errors='coerce')
            suspicious_aum = df[(df['aum'] > 500000) & df['aum'].notna()]
            if len(suspicious_aum) > 0:
                st.warning(f"⚠️ Found {len(suspicious_aum)} funds with AUM > $500B in {filename}")
                with st.expander("🔍 View Large AUM Funds"):
                    st.dataframe(suspicious_aum[['fund_name', 'aum']], use_container_width=True)
        
        # Clean strategies
        if 'strategy' in df.columns:
            df['strategy'] = df['strategy'].astype(str).str.strip()
            df.loc[df['strategy'].isin(['None', 'nan', 'null', '']) | df['strategy'].isnull(), 'strategy'] = 'Multi-Strategy'
        
        # Remove duplicate fund names (keep first occurrence)
        df = df.drop_duplicates(subset=['fund_name'], keep='first')
        
        final_count = len(df)
        if final_count < original_count:
            st.info(f"🧹 Data cleaning: {original_count} → {final_count} funds in {filename}")
        
        return df

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the extracted DataFrame with enhanced validation"""
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
        
        # Clean and standardize strategy names
        if 'strategy' in df.columns:
            df['strategy'] = df['strategy'].astype(str).str.strip()
            
            # Handle null/None/nan strategies
            df.loc[df['strategy'].isin(['None', 'nan', 'null', '']) | df['strategy'].isnull(), 'strategy'] = 'Multi-Strategy'
            
            # Standardize common strategy name variations
            strategy_mapping = {
                'L/S Equity': 'Long/Short Equity',
                'L/S Eq': 'Long/Short Equity',
                'L/S': 'Long/Short Equity',
                'Long/Short Eq': 'Long/Short Equity',
                'Fixed Income Arb': 'Fixed Income Arbitrage',
                'EventDriven': 'Event-Driven',
                'Quant': 'Quantitative',
                'Vol': 'Volatility',
                'Commodity': 'Commodities'
            }
            
            for old_name, new_name in strategy_mapping.items():
                df.loc[df['strategy'] == old_name, 'strategy'] = new_name
        
        # Calculate net return in USD if both AUM and return are available
        if 'aum' in df.columns and 'return' in df.columns:
            df['net_return_usd'] = df['return'] * df['aum']
        
        # Remove rows with missing critical data (fund name is essential)
        df = df.dropna(subset=['fund_name'])
        
        return df

    # Enhanced visualization functions with dark theme styling

def create_visualizations(df: pd.DataFrame):
    """Create intuitive and meaningful fund analysis visualizations with dark theme"""
    if df.empty:
        st.warning("📊 No data available for visualization")
        return
    
    st.markdown("## 📊 Fund Performance Dashboard")
    st.markdown("Comprehensive analysis and insights from your fund data")
    
    # Enhanced tabs with professional icons
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Performance Ranking", 
        "💰 Capital Analysis", 
        "📈 Strategy Insights", 
        "⚠️ Risk Assessment"
    ])
    
    with tab1:
        st.markdown("### 🏆 Fund Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.markdown("**🏆 Fund Performance Leaderboard**")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('#1e2329')
                ax.set_facecolor('#263238')
                
                # Sort funds by performance
                sorted_df = df.sort_values('return', ascending=True)
                fund_names = [name[:25] + "..." if len(name) > 25 else name for name in sorted_df['fund_name']]
                returns = sorted_df['return'].values
                
                # Enhanced color scheme for dark theme
                colors = []
                for ret in returns:
                    if ret > 0.005:  # > 0.5%
                        colors.append('#4caf50')  # Green
                    elif ret > 0:
                        colors.append('#81c784')  # Light Green
                    elif ret > -0.005:  # > -0.5%
                        colors.append('#ffb74d')  # Orange
                    else:
                        colors.append('#f44336')  # Red
                
                bars = ax.barh(range(len(returns)), returns, color=colors, alpha=0.8, edgecolor='#404040', linewidth=0.5)
                ax.set_yticks(range(len(returns)))
                ax.set_yticklabels(fund_names, fontsize=10, color='#e8eaed')
                ax.set_xlabel('Return (%)', fontsize=12, fontweight='bold', color='#e8eaed')
                ax.set_title('Fund Performance Ranking', fontsize=14, fontweight='bold', pad=20, color='#ffffff')
                
                # Enhanced value labels
                for i, v in enumerate(returns):
                    label_color = 'white' if abs(v) > 0.003 else '#263238'
                    ax.text(v/2, i, f'{v:.2%}', va='center', ha='center', 
                           fontweight='bold', color=label_color, fontsize=9)
                
                # Enhanced styling
                ax.axvline(x=0, color='#64b5f6', linestyle='--', alpha=0.8, linewidth=2)
                ax.grid(axis='x', alpha=0.2, color='#404040')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#404040')
                ax.spines['left'].set_color('#404040')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'return' in df.columns and 'strategy' in df.columns:
                st.markdown("**📊 Performance by Strategy Distribution**")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('#1e2329')
                ax.set_facecolor('#263238')
                
                # Create strategy performance box plot
                strategy_data = []
                strategy_labels = []
                
                for strategy in df['strategy'].unique():
                    if pd.notna(strategy):
                        strategy_returns = df[df['strategy'] == strategy]['return'].dropna()
                        if len(strategy_returns) > 0:
                            strategy_data.append(strategy_returns.values)
                            strategy_labels.append(strategy[:20])
                
                if strategy_data:
                    bp = ax.boxplot(strategy_data, labels=strategy_labels, patch_artist=True,
                                   boxprops=dict(facecolor='#42a5f5', alpha=0.7),
                                   medianprops=dict(color='#ffffff', linewidth=2),
                                   whiskerprops=dict(color='#90a4ae'),
                                   capprops=dict(color='#90a4ae'))
                    
                    # Enhanced color scheme
                    colors = ['#42a5f5', '#66bb6a', '#ff7043', '#ab47bc', '#26c6da', '#ffa726', '#ec407a']
                    for i, patch in enumerate(bp['boxes']):
                        patch.set_facecolor(colors[i % len(colors)])
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold', color='#e8eaed')
                    ax.set_title('Return Distribution by Strategy', fontsize=14, fontweight='bold', pad=20, color='#ffffff')
                    ax.tick_params(axis='x', rotation=45, colors='#e8eaed')
                    ax.tick_params(axis='y', colors='#e8eaed')
                    ax.grid(axis='y', alpha=0.2, color='#404040')
                    ax.axhline(y=0, color='#f44336', linestyle='--', alpha=0.8, linewidth=2)
                    
                    # Enhanced styling
                    for spine in ax.spines.values():
                        spine.set_color('#404040')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        st.markdown("### 💰 Capital Allocation Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'aum' in df.columns and not df['aum'].isnull().all():
                st.markdown("**💰 Capital Allocation Overview**")
                
                fig, ax = plt.subplots(figsize=(10, 10))
                fig.patch.set_facecolor('#1e2329')
                
                strategy_aum = df.groupby('strategy')['aum'].sum().sort_values(ascending=False)
                
                # Professional color palette for dark theme
                colors = ['#42a5f5', '#66bb6a', '#ff7043', '#ab47bc', '#26c6da', '#ffa726', '#ec407a']
                
                wedges, texts, autotexts = ax.pie(strategy_aum.values, 
                                                 labels=strategy_aum.index,
                                                 autopct=lambda pct: f'${strategy_aum.sum()*pct/100:.0f}M\n({pct:.1f}%)',
                                                 colors=colors[:len(strategy_aum)],
                                                 startangle=90,
                                                 textprops={'fontsize': 10, 'color': '#e8eaed'},
                                                 wedgeprops=dict(edgecolor='#404040', linewidth=1))
                
                # Enhanced text styling - FIXED VERSION
                for autotext in autotexts:
                    autotext.set_color('#ffffff')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                    # Use tuple format instead of rgba string
                    autotext.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=(0,0,0,0.7), edgecolor='none'))
                
                for text in texts:
                    text.set_fontweight('bold')
                    text.set_fontsize(11)
                
                ax.set_title('AUM Distribution by Strategy', fontsize=16, fontweight='bold', pad=30, color='#ffffff')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'aum' in df.columns and 'return' in df.columns:
                st.markdown("**📈 Capital vs Performance Analysis**")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('#1e2329')
                ax.set_facecolor('#263238')
                
                # Enhanced scatter plot
                scatter = ax.scatter(df['return'], df['aum'], 
                                   s=120, alpha=0.8, 
                                   c=range(len(df)), cmap='viridis',
                                   edgecolors='#404040', linewidth=1)
                
                # Enhanced fund labels with better positioning - FIXED VERSION
                for i, row in df.iterrows():
                    if pd.notna(row['return']) and pd.notna(row['aum']):
                        label = row['fund_name'][:20] + "..." if len(row['fund_name']) > 20 else row['fund_name']
                        ax.annotate(label, (row['return'], row['aum']), 
                                   xytext=(8, 8), textcoords='offset points',
                                   fontsize=8, alpha=0.9, color='#e8eaed',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=(0,0,0,0.7), edgecolor='none'))
                
                ax.set_xlabel('Return (%)', fontsize=12, fontweight='bold', color='#e8eaed')
                ax.set_ylabel('AUM ($M)', fontsize=12, fontweight='bold', color='#e8eaed')
                ax.set_title('Fund Size vs Performance', fontsize=14, fontweight='bold', pad=20, color='#ffffff')
                ax.grid(True, alpha=0.2, color='#404040')
                ax.axvline(x=0, color='#f44336', linestyle='--', alpha=0.8, linewidth=2)
                
                # Enhanced styling
                for spine in ax.spines.values():
                    spine.set_color('#404040')
                ax.tick_params(colors='#e8eaed')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        st.markdown("### 📈 Strategy Performance Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'strategy' in df.columns and 'return' in df.columns:
                st.markdown("**🎯 Strategy Performance Comparison**")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('#1e2329')
                ax.set_facecolor('#263238')
                
                strategy_stats = df.groupby('strategy').agg({
                    'return': ['mean', 'count'],
                    'aum': 'sum'
                }).round(4)
                
                strategy_stats.columns = ['Avg_Return', 'Fund_Count', 'Total_AUM']
                strategy_stats = strategy_stats.sort_values('Avg_Return', ascending=True)
                
                # Enhanced color coding
                colors = ['#f44336' if x < 0 else '#ff9800' if x < 0.005 else '#4caf50' 
                         for x in strategy_stats['Avg_Return']]
                
                bars = ax.barh(range(len(strategy_stats)), strategy_stats['Avg_Return'], 
                              color=colors, alpha=0.8, edgecolor='#404040', linewidth=0.5)
                
                ax.set_yticks(range(len(strategy_stats)))
                ax.set_yticklabels(strategy_stats.index, fontsize=11, color='#e8eaed')
                ax.set_xlabel('Average Return (%)', fontsize=12, fontweight='bold', color='#e8eaed')
                ax.set_title('Strategy Performance Ranking', fontsize=14, fontweight='bold', pad=20, color='#ffffff')
                
                # Enhanced value labels - FIXED VERSION
                for i, (ret, count) in enumerate(zip(strategy_stats['Avg_Return'], strategy_stats['Fund_Count'])):
                    label_pos = ret + 0.001 if ret > 0 else ret - 0.001
                    ha = 'left' if ret > 0 else 'right'
                    ax.text(label_pos, i, f'{ret:.2%} ({int(count)} funds)', 
                           va='center', ha=ha, fontsize=10, color='#e8eaed',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=(0,0,0,0.7), edgecolor='none'))
                
                ax.axvline(x=0, color='#64b5f6', linestyle='--', alpha=0.8, linewidth=2)
                ax.grid(axis='x', alpha=0.2, color='#404040')
                
                # Enhanced styling
                for spine in ax.spines.values():
                    spine.set_color('#404040')
                ax.tick_params(colors='#e8eaed')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'strategy' in df.columns:
                st.markdown("**📋 Strategy Summary Table**")
                
                strategy_summary = df.groupby('strategy').agg({
                    'return': ['mean', 'std', 'min', 'max', 'count'],
                    'aum': ['sum', 'mean']
                }).round(4)
                
                strategy_summary.columns = ['Avg Return', 'Volatility', 'Min Return', 'Max Return', 
                                          'Funds', 'Total AUM', 'Avg Fund Size']
                
                # Format for display with enhanced styling
                display_summary = strategy_summary.copy()
                for col in ['Avg Return', 'Volatility', 'Min Return', 'Max Return']:
                    display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                
                for col in ['Total AUM', 'Avg Fund Size']:
                    display_summary[col] = display_summary[col].apply(lambda x: f"${x:.0f}M" if pd.notna(x) else "N/A")
                
                display_summary['Funds'] = display_summary['Funds'].astype(int)
                
                # Enhanced dataframe display
                st.dataframe(
                    display_summary, 
                    use_container_width=True,
                    height=400
                )
    
    with tab4:
        st.markdown("### ⚠️ Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.markdown("**⚠️ Risk Distribution Analysis**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_facecolor('#1e2329')
                ax.set_facecolor('#263238')
                
                returns_clean = df['return'].dropna()
                
                # Define risk categories
                high_risk = returns_clean[abs(returns_clean) > 0.01].count()
                medium_risk = returns_clean[(abs(returns_clean) > 0.005) & (abs(returns_clean) <= 0.01)].count()
                low_risk = returns_clean[abs(returns_clean) <= 0.005].count()
                
                categories = ['Low Risk\n(±0.5%)', 'Medium Risk\n(±0.5-1%)', 'High Risk\n(>±1%)']
                counts = [low_risk, medium_risk, high_risk]
                colors = ['#4caf50', '#ff9800', '#f44336']
                
                bars = ax.bar(categories, counts, color=colors, alpha=0.8, 
                             edgecolor='#404040', linewidth=1)
                ax.set_ylabel('Number of Funds', fontsize=12, fontweight='bold', color='#e8eaed')
                ax.set_title('Risk Profile Distribution', fontsize=14, fontweight='bold', pad=20, color='#ffffff')
                
                # Enhanced count labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold',
                           fontsize=12, color='#e8eaed')
                
                ax.grid(axis='y', alpha=0.2, color='#404040')
                for spine in ax.spines.values():
                    spine.set_color('#404040')
                ax.tick_params(colors='#e8eaed')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.markdown("**📊 Portfolio Risk Metrics**")
                
                returns_clean = df['return'].dropna()
                
                if len(returns_clean) > 0:
                    # Calculate comprehensive risk metrics
                    portfolio_return = returns_clean.mean()
                    portfolio_vol = returns_clean.std()
                    downside_deviation = returns_clean[returns_clean < 0].std() if len(returns_clean[returns_clean < 0]) > 0 else 0
                    max_drawdown = returns_clean.min()
                    win_rate = (returns_clean > 0).sum() / len(returns_clean)
                    
                    # Enhanced metrics display with professional styling
                    metrics_data = {
                        'Metric': ['Portfolio Return', 'Volatility', 'Downside Risk', 'Max Drawdown', 
                                 'Win Rate', 'Best Performer', 'Worst Performer'],
                        'Value': [
                            f'{portfolio_return:.2%}',
                            f'{portfolio_vol:.2%}',
                            f'{downside_deviation:.2%}',
                            f'{max_drawdown:.2%}',
                            f'{win_rate:.1%}',
                            f'{returns_clean.max():.2%}',
                            f'{returns_clean.min():.2%}'
                        ],
                        'Status': [
                            '🟢' if portfolio_return > 0 else '🔴',
                            '🟢' if portfolio_vol < 0.01 else '🟡' if portfolio_vol < 0.02 else '🔴',
                            '🟢' if downside_deviation < 0.01 else '🟡' if downside_deviation < 0.02 else '🔴',
                            '🔴' if max_drawdown < -0.005 else '🟡' if max_drawdown < 0 else '🟢',
                            '🟢' if win_rate > 0.7 else '🟡' if win_rate > 0.5 else '🔴',
                            '🟢',
                            '🔴' if max_drawdown < -0.005 else '🟡'
                        ]
                    }
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True, height=280)

# def create_download_links(df: pd.DataFrame):
#     """Create enhanced download links for reports with professional styling"""
    
#     st.markdown("""
#     <div class="analysis-section">
#         <h2>⬇️ Download Professional Reports</h2>
#         <p style="color: #90a4ae; margin-bottom: 1rem;">
#             Export your analysis in multiple formats for further use
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("**📊 Excel Report**")
#         st.markdown("Comprehensive analysis with multiple sheets")
        
#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             df.to_excel(writer, index=False, sheet_name='Fund Analysis')
            
#             if 'strategy' in df.columns:
#                 summary = df.groupby('strategy').agg({
#                     'return': ['mean', 'std', 'count'],
#                     'aum': 'sum'
#                 }).round(4)
#                 summary.to_excel(writer, sheet_name='Strategy Summary')
        
#         excel_data = output.getvalue()
#         b64 = base64.b64encode(excel_data).decode()
#         href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="fund_analysis.xlsx" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #42a5f5, #2196f3); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; width: 100%; margin-top: 10px;">📊 Download Excel</button></a>'
#         st.markdown(href, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("**📋 CSV Data**")
#         st.markdown("Raw data for further analysis")
        
#         csv_data = df.to_csv(index=False)
#         b64 = base64.b64encode(csv_data.encode()).decode()
#         href = f'<a href="data:text/csv;base64,{b64}" download="fund_data.csv" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #66bb6a, #4caf50); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; width: 100%; margin-top: 10px;">📋 Download CSV</button></a>'
#         st.markdown(href, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("**🔧 JSON Data**")
#         st.markdown("Structured data for developers")
        
#         json_data = df.to_json(orient='records', indent=2)
#         b64 = base64.b64encode(json_data.encode()).decode()
#         href = f'<a href="data:application/json;base64,{b64}" download="fund_data.json" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #ab47bc, #9c27b0); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; width: 100%; margin-top: 10px;">🔧 Download JSON</button></a>'
#         st.markdown(href, unsafe_allow_html=True)

def generate_executive_summary_pdf(df: pd.DataFrame) -> bytes:
    """Generate a professional multi-page executive summary PDF with proper spacing"""
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime
    
    # Create PDF
    pdf_buffer = BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        
        # === PAGE 1: EXECUTIVE SUMMARY ===
        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 size
        fig1.patch.set_facecolor('white')
        
        # Title with proper spacing
        fig1.text(0.5, 0.95, 'FUND PERFORMANCE EXECUTIVE SUMMARY', 
                 fontsize=16, fontweight='bold', ha='center')
        
        # Header details with NO overlapping - fixed positioning
        report_date = datetime.now().strftime('%B %d, %Y')
        total_funds = len(df)
        total_strategies = df["strategy"].nunique() if "strategy" in df.columns else 0
        
        # Row 1: Date and Portfolio info with proper spacing
        fig1.text(0.1, 0.89, f'Report Date: {report_date}', fontsize=10, fontweight='bold')
        fig1.text(0.1, 0.86, f'Portfolio Overview: {total_funds} Funds across {total_strategies} Strategies', 
                 fontsize=10, fontweight='bold')
        
        # Row 2: AUM on separate line to avoid overlap
        if 'aum' in df.columns and not df['aum'].isnull().all():
            total_aum = df['aum'].sum()
            fig1.text(0.1, 0.83, f'Total AUM: ${total_aum:,.0f}M', fontsize=10, fontweight='bold')
        
        # Separator line with more space
        fig1.add_artist(plt.Line2D([0.1, 0.9], [0.80, 0.80], color='#333', linewidth=1))
        
        # === KEY METRICS SECTION ===
        if 'return' in df.columns and not df['return'].isnull().all():
            fig1.text(0.1, 0.76, 'KEY PERFORMANCE METRICS', fontsize=12, fontweight='bold')
            
            # Calculate metrics
            avg_return = df['return'].mean()
            best_return = df['return'].max()
            worst_return = df['return'].min()
            win_rate = (df['return'] > 0).sum() / len(df[df['return'].notna()])
            volatility = df['return'].std()
            
            # Metrics in grid format with proper spacing
            metrics_y = 0.70
            
            # Row 1 - Portfolio metrics
            fig1.text(0.1, metrics_y, 'Portfolio Average Return:', fontsize=10, fontweight='bold')
            color = 'green' if avg_return > 0 else 'red'
            fig1.text(0.35, metrics_y, f'{avg_return:.2%}', fontsize=10, color=color, fontweight='bold')
            
            fig1.text(0.5, metrics_y, 'Best Performer:', fontsize=10, fontweight='bold')
            fig1.text(0.7, metrics_y, f'{best_return:.2%}', fontsize=10, color='green', fontweight='bold')
            
            # Row 2 - Risk metrics
            metrics_y -= 0.035
            fig1.text(0.1, metrics_y, 'Portfolio Volatility:', fontsize=10, fontweight='bold')
            vol_color = 'green' if volatility < 0.01 else 'orange' if volatility < 0.02 else 'red'
            fig1.text(0.35, metrics_y, f'{volatility:.2%}', fontsize=10, color=vol_color, fontweight='bold')
            
            fig1.text(0.5, metrics_y, 'Win Rate:', fontsize=10, fontweight='bold')
            win_color = 'green' if win_rate > 0.6 else 'orange' if win_rate > 0.4 else 'red'
            fig1.text(0.7, metrics_y, f'{win_rate:.1%}', fontsize=10, color=win_color, fontweight='bold')
            
            # Row 3 - Additional metrics
            metrics_y -= 0.035
            fig1.text(0.1, metrics_y, 'Return Range:', fontsize=10, fontweight='bold')
            fig1.text(0.35, metrics_y, f'{worst_return:.2%} to {best_return:.2%}', fontsize=10)
            
            if 'aum' in df.columns and not df['aum'].isnull().all():
                avg_fund_size = df['aum'].mean()
                fig1.text(0.5, metrics_y, 'Average Fund Size:', fontsize=10, fontweight='bold')
                fig1.text(0.7, metrics_y, f'${avg_fund_size:.0f}M', fontsize=10)
        
        # === TOP PERFORMERS TABLE ===
        if 'return' in df.columns and not df['return'].isnull().all():
            table_y = 0.58
            fig1.text(0.1, table_y, 'TOP 5 PERFORMING FUNDS', fontsize=12, fontweight='bold')
            
            # Get top 5 performers
            top_performers = df.nlargest(5, 'return')[['fund_name', 'return', 'strategy', 'aum']]
            
            # Table headers with perfect alignment
            header_y = table_y - 0.04
            fig1.text(0.1, header_y, 'Fund Name', fontsize=9, fontweight='bold')
            fig1.text(0.45, header_y, 'Return', fontsize=9, fontweight='bold')
            fig1.text(0.58, header_y, 'Strategy', fontsize=9, fontweight='bold')
            if 'aum' in df.columns:
                fig1.text(0.8, header_y, 'AUM ($M)', fontsize=9, fontweight='bold')
            
            # Add header underline
            fig1.add_artist(plt.Line2D([0.1, 0.9], [header_y - 0.008, header_y - 0.008], color='gray', linewidth=0.5))
            
            # Table data with consistent spacing
            for i, (_, row) in enumerate(top_performers.iterrows()):
                y = header_y - 0.03 - (i * 0.025)
                
                # Fund name (truncated properly)
                fund_name = row['fund_name'][:28] + "..." if len(row['fund_name']) > 28 else row['fund_name']
                fig1.text(0.1, y, fund_name, fontsize=8)
                
                # Return with color coding
                ret_color = 'green' if row['return'] > 0 else 'red'
                fig1.text(0.45, y, f"{row['return']:.2%}", fontsize=8, color=ret_color, fontweight='bold')
                
                # Strategy (truncated properly)
                strategy = str(row['strategy'])[:18] + "..." if len(str(row['strategy'])) > 18 else str(row['strategy'])
                fig1.text(0.58, y, strategy, fontsize=8)
                
                # AUM if available
                if 'aum' in df.columns and pd.notna(row['aum']):
                    fig1.text(0.8, y, f"${row['aum']:.0f}", fontsize=8)
        
        # === STRATEGY ANALYSIS ===
        if 'strategy' in df.columns and 'return' in df.columns:
            strategy_y = 0.35
            fig1.text(0.1, strategy_y, 'STRATEGY PERFORMANCE ANALYSIS', fontsize=12, fontweight='bold')
            
            # Strategy summary
            strategy_stats = df.groupby('strategy').agg({
                'return': ['mean', 'std', 'count'],
                'aum': 'sum'
            }).round(4)
            
            strategy_stats.columns = ['Avg_Return', 'Volatility', 'Fund_Count', 'Total_AUM']
            strategy_stats = strategy_stats.sort_values('Avg_Return', ascending=False)
            
            # Best and worst strategies with proper spacing
            best_strategy = strategy_stats.index[0]
            best_strategy_return = strategy_stats.iloc[0]['Avg_Return']
            worst_strategy = strategy_stats.index[-1]
            worst_strategy_return = strategy_stats.iloc[-1]['Avg_Return']
            
            summary_y = strategy_y - 0.04
            fig1.text(0.1, summary_y, f'Best Performing Strategy:', fontsize=10, fontweight='bold')
            fig1.text(0.4, summary_y, f'{best_strategy} ({best_strategy_return:.2%})', 
                     fontsize=10, color='green', fontweight='bold')
            
            summary_y -= 0.03
            fig1.text(0.1, summary_y, f'Lowest Performing Strategy:', fontsize=10, fontweight='bold')
            fig1.text(0.4, summary_y, f'{worst_strategy} ({worst_strategy_return:.2%})', 
                     fontsize=10, color='red', fontweight='bold')
            
            # Strategy diversification
            total_strategies = len(strategy_stats)
            most_funds_strategy = strategy_stats.loc[strategy_stats['Fund_Count'].idxmax()]
            
            summary_y -= 0.03
            fig1.text(0.1, summary_y, f'Total Investment Strategies: {total_strategies}', fontsize=10)
            
            summary_y -= 0.03
            fig1.text(0.1, summary_y, f'Most Diversified Strategy:', fontsize=10, fontweight='bold')
            fig1.text(0.4, summary_y, f'{most_funds_strategy.name} ({int(most_funds_strategy["Fund_Count"])} funds)', 
                     fontsize=10)
        
        # === RISK ASSESSMENT ===
        if 'return' in df.columns and not df['return'].isnull().all():
            risk_y = 0.15
            fig1.text(0.1, risk_y, 'RISK ASSESSMENT', fontsize=12, fontweight='bold')
            
            returns_clean = df['return'].dropna()
            
            # Risk metrics with proper alignment
            downside_returns = returns_clean[returns_clean < 0]
            upside_returns = returns_clean[returns_clean > 0]
            
            risk_summary_y = risk_y - 0.04
            
            fig1.text(0.1, risk_summary_y, f'Funds with Positive Returns:', fontsize=10, fontweight='bold')
            fig1.text(0.4, risk_summary_y, f'{len(upside_returns)}/{len(returns_clean)} ({len(upside_returns)/len(returns_clean):.1%})', 
                     fontsize=10, color='green')
            
            if len(downside_returns) > 0:
                avg_downside = downside_returns.mean()
                risk_summary_y -= 0.03
                fig1.text(0.1, risk_summary_y, f'Average Downside:', fontsize=10, fontweight='bold')
                fig1.text(0.4, risk_summary_y, f'{avg_downside:.2%}', fontsize=10, color='red')
            
            if len(upside_returns) > 0:
                avg_upside = upside_returns.mean()
                risk_summary_y -= 0.03
                fig1.text(0.1, risk_summary_y, f'Average Upside:', fontsize=10, fontweight='bold')
                fig1.text(0.4, risk_summary_y, f'{avg_upside:.2%}', fontsize=10, color='green')
            
            # Risk categorization
            risk_level = "Low" if volatility < 0.005 else "Medium" if volatility < 0.015 else "High"
            risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
            risk_summary_y -= 0.03
            fig1.text(0.1, risk_summary_y, f'Overall Portfolio Risk Level:', fontsize=10, fontweight='bold')
            fig1.text(0.4, risk_summary_y, risk_level, fontsize=10, color=risk_color, fontweight='bold')
        
        pdf.savefig(fig1, bbox_inches='tight', dpi=300)
        plt.close(fig1)
        
        # === PAGE 2: PERFORMANCE CHARTS ===
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.patch.set_facecolor('white')
        
        # Page 2 Title
        # fig2.text(0.5, 0.95, 'PERFORMANCE ANALYSIS CHARTS', 
        #          fontsize=16, fontweight='bold', ha='center')
        
        # Chart 1: Fund Performance Ranking - FIXED SPACING
        if 'return' in df.columns and not df['return'].isnull().all():
            ax1 = fig2.add_subplot(2, 2, 1)
            
            # Sort all funds by performance
            sorted_df = df.sort_values('return', ascending=True)
            fund_names = [name[:12] + "..." if len(name) > 12 else name for name in sorted_df['fund_name']]
            returns = sorted_df['return'].values
            
            # Color coding
            colors = ['green' if r > 0 else 'red' for r in returns]
            
            bars = ax1.barh(range(len(returns)), returns, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(returns)))
            ax1.set_yticklabels(fund_names, fontsize=7)
            ax1.set_xlabel('Return (%)', fontsize=9, fontweight='bold')
            ax1.set_title('Fund Performance Ranking', fontsize=10, fontweight='bold', pad=10)
            ax1.grid(axis='x', alpha=0.3)
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            
            # Add value labels with better positioning to avoid overlap
            for i, v in enumerate(returns):
                if v > 0:
                    label_pos = v * 0.5  # Position at 50% of bar length
                    ha = 'center'
                    color = 'white'
                else:
                    label_pos = v * 0.5  # Position at 50% of bar length
                    ha = 'center'
                    color = 'white'
                ax1.text(label_pos, i, f'{v:.1%}', 
                        va='center', ha=ha, fontsize=6, fontweight='bold', color=color)
        
        # Chart 2: Strategy Performance Comparison - FIXED SPACING
        if 'strategy' in df.columns and 'return' in df.columns:
            ax2 = fig2.add_subplot(2, 2, 2)
            
            strategy_returns = df.groupby('strategy')['return'].mean().sort_values(ascending=True)
            
            colors = ['red' if r < 0 else 'lightgreen' if r < 0.005 else 'green' 
                     for r in strategy_returns.values]
            
            bars = ax2.barh(range(len(strategy_returns)), strategy_returns.values, color=colors, alpha=0.8)
            ax2.set_yticks(range(len(strategy_returns)))
            ax2.set_yticklabels([s[:10] + "..." if len(s) > 10 else s for s in strategy_returns.index], fontsize=7)
            ax2.set_xlabel('Average Return (%)', fontsize=9, fontweight='bold')
            ax2.set_title('Strategy Performance', fontsize=10, fontweight='bold', pad=10)
            ax2.grid(axis='x', alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            
            # Add value labels with better positioning
            for i, v in enumerate(strategy_returns.values):
                if v > 0:
                    label_pos = v * 0.5
                    ha = 'center'
                    color = 'white'
                else:
                    label_pos = v * 0.5
                    ha = 'center'
                    color = 'white'
                ax2.text(label_pos, i, f'{v:.1%}', 
                        va='center', ha=ha, fontsize=6, fontweight='bold', color=color)
        
        # Chart 3: AUM Distribution by Strategy - CLEANER PIE CHART
        if 'aum' in df.columns and 'strategy' in df.columns and not df['aum'].isnull().all():
            ax3 = fig2.add_subplot(2, 2, 3)
            
            strategy_aum = df.groupby('strategy')['aum'].sum().sort_values(ascending=False)
            
            # Limit to top strategies for readability
            if len(strategy_aum) > 6:
                top_strategies = strategy_aum.head(5)
                other_aum = strategy_aum.tail(-5).sum()
                if other_aum > 0:
                    top_strategies['Other'] = other_aum
                strategy_aum = top_strategies
            
            colors = plt.cm.Set3(range(len(strategy_aum)))
            
            # Create pie chart with better label positioning
            wedges, texts, autotexts = ax3.pie(strategy_aum.values, 
                                              labels=None,  # Remove labels to avoid overlap
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              startangle=90,
                                              pctdistance=0.85)
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(7)
            
            # Add legend instead of labels
            ax3.legend(wedges, [s[:8] + "..." if len(s) > 8 else s for s in strategy_aum.index],
                      title="Strategies", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=7)
            
            ax3.set_title('AUM Distribution', fontsize=10, fontweight='bold', pad=10)
        
        # Chart 4: Risk vs Return Scatter Plot - CLEANER VERSION
        if 'return' in df.columns and 'aum' in df.columns:
            ax4 = fig2.add_subplot(2, 2, 4)
            
            # Create scatter plot
            scatter = ax4.scatter(df['return'], df['aum'], 
                                s=60, alpha=0.7, c=range(len(df)), cmap='viridis')
            
            ax4.set_xlabel('Return (%)', fontsize=9, fontweight='bold')
            ax4.set_ylabel('AUM ($M)', fontsize=9, fontweight='bold')
            ax4.set_title('Fund Size vs Performance', fontsize=10, fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3)
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            
            # Simplified quadrant labels (no overlap)
            max_aum = df['aum'].max() if not df['aum'].isnull().all() else 1000
            max_ret = df['return'].max() if not df['return'].isnull().all() else 0.01
            min_ret = df['return'].min() if not df['return'].isnull().all() else -0.01
            
            # Only add quadrant labels if there's space
            if max_ret > 0.005:
                ax4.text(max_ret * 0.8, max_aum * 0.9, 'High Return\nLarge AUM', 
                        fontsize=7, ha='center', va='center', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.6))
            
            if min_ret < -0.001:
                ax4.text(min_ret * 0.8, max_aum * 0.9, 'Low Return\nLarge AUM', 
                        fontsize=7, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.6))
        
        plt.tight_layout(pad=3.0)  # Add more padding between subplots
        
        pdf.savefig(fig2, bbox_inches='tight', dpi=300)
        plt.close(fig2)
        
        # === PAGE 3: DETAILED STRATEGY BREAKDOWN ===
        fig3 = plt.figure(figsize=(8.27, 11.69))
        fig3.patch.set_facecolor('white')
        
        # Page 3 Title
        fig3.text(0.5, 0.95, 'DETAILED STRATEGY BREAKDOWN', 
                 fontsize=16, fontweight='bold', ha='center')
        
        if 'strategy' in df.columns:
            # Strategy summary table
            table_y = 0.85
            fig3.text(0.1, table_y, 'COMPREHENSIVE STRATEGY ANALYSIS', fontsize=12, fontweight='bold')
            
            # Calculate detailed strategy metrics
            strategy_detailed = df.groupby('strategy').agg({
                'return': ['mean', 'std', 'min', 'max', 'count'],
                'aum': ['sum', 'mean']
            }).round(4)
            
            strategy_detailed.columns = ['Avg_Return', 'Volatility', 'Min_Return', 'Max_Return', 
                                       'Fund_Count', 'Total_AUM', 'Avg_Fund_Size']
            
            # Table headers with perfect spacing
            header_y = table_y - 0.06
            headers = ['Strategy', 'Avg Return', 'Volatility', 'Funds', 'Total AUM', 'Best', 'Worst']
            positions = [0.1, 0.28, 0.42, 0.54, 0.64, 0.78, 0.86]
            
            for header, pos in zip(headers, positions):
                fig3.text(pos, header_y, header, fontsize=8, fontweight='bold')
            
            # Add header underline
            fig3.add_artist(plt.Line2D([0.1, 0.92], [header_y - 0.008, header_y - 0.008], color='gray', linewidth=0.5))
            
            # Table data with consistent spacing
            for i, (strategy, row) in enumerate(strategy_detailed.iterrows()):
                y = header_y - 0.03 - (i * 0.025)
                
                # Strategy name (truncated)
                strategy_name = strategy[:12] + "..." if len(strategy) > 12 else strategy
                fig3.text(0.1, y, strategy_name, fontsize=7)
                
                # Average return with color
                avg_ret_color = 'green' if row['Avg_Return'] > 0 else 'red'
                fig3.text(0.28, y, f"{row['Avg_Return']:.2%}", fontsize=7, color=avg_ret_color, fontweight='bold')
                
                # Volatility (handle NaN properly)
                if pd.notna(row['Volatility']):
                    fig3.text(0.42, y, f"{row['Volatility']:.2%}", fontsize=7)
                else:
                    fig3.text(0.42, y, "N/A", fontsize=7, alpha=0.6)
                
                # Fund count
                fig3.text(0.54, y, f"{int(row['Fund_Count'])}", fontsize=7)
                
                # Total AUM
                if pd.notna(row['Total_AUM']):
                    fig3.text(0.64, y, f"${row['Total_AUM']:.0f}M", fontsize=7)
                
                # Best and worst returns
                best_color = 'green' if row['Max_Return'] > 0 else 'red'
                worst_color = 'green' if row['Min_Return'] > 0 else 'red'
                fig3.text(0.78, y, f"{row['Max_Return']:.2%}", fontsize=7, color=best_color)
                fig3.text(0.86, y, f"{row['Min_Return']:.2%}", fontsize=7, color=worst_color)
        
        # Investment insights section with better spacing
        if 'return' in df.columns and 'strategy' in df.columns:
            insights_y = 0.45
            fig3.text(0.1, insights_y, 'INVESTMENT INSIGHTS', fontsize=12, fontweight='bold')
            
            # Calculate insights
            best_strategy = df.groupby('strategy')['return'].mean().idxmax()
            best_strategy_return = df.groupby('strategy')['return'].mean().max()
            
            most_consistent = df.groupby('strategy')['return'].std().idxmin()
            most_consistent_vol = df.groupby('strategy')['return'].std().min()
            
            insight_text_y = insights_y - 0.05
            
            # Key insights with proper bullet points and spacing
            insights = [
                f"🏆 Best Performing Strategy: {best_strategy} (Average: {best_strategy_return:.2%})",
                f"📊 Most Consistent Strategy: {most_consistent} (Volatility: {most_consistent_vol:.2%})",
                f"💰 Total Portfolio Value: ${df['aum'].sum():.0f}M across {len(df)} funds",
                f"⚡ Portfolio Momentum: {(df['return'] > 0).sum()}/{len(df)} funds showing positive returns"
            ]
            
            for i, insight in enumerate(insights):
                fig3.text(0.1, insight_text_y - i*0.035, insight, fontsize=10)
        
        pdf.savefig(fig3, bbox_inches='tight', dpi=300)
        plt.close(fig3)
    
    return pdf_buffer.getvalue()

def create_download_links(df: pd.DataFrame):
    """Create enhanced download links for reports with professional styling including Executive Summary PDF"""
    
    st.markdown("## ⬇️ Download Professional Reports")
    st.markdown("Export your analysis in multiple formats for further use")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📄 Executive Summary**")
        st.markdown("Professional PDF report")
        
        if st.button("📄 Generate Executive Summary", use_container_width=True):
            with st.spinner("Creating executive summary..."):
                try:
                    pdf_bytes = generate_executive_summary_pdf(df)
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="executive_summary.pdf" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #e91e63, #ad1457); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; width: 100%; margin-top: 10px;">📄 Download Executive PDF</button></a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Executive summary generated!")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
    
    with col2:
        st.markdown("**📊 Excel Report**")
        st.markdown("Comprehensive analysis with multiple sheets")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Fund Analysis')
            
            if 'strategy' in df.columns:
                summary = df.groupby('strategy').agg({
                    'return': ['mean', 'std', 'count'],
                    'aum': 'sum'
                }).round(4)
                summary.to_excel(writer, sheet_name='Strategy Summary')
        
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="fund_analysis.xlsx" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #42a5f5, #2196f3); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; width: 100%; margin-top: 10px;">📊 Download Excel</button></a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**📋 CSV Data**")
        st.markdown("Raw data for further analysis")
        
        csv_data = df.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="fund_data.csv" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #66bb6a, #4caf50); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; width: 100%; margin-top: 10px;">📋 Download CSV</button></a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col4:
        st.markdown("**🔧 JSON Data**")
        st.markdown("Structured data for developers")
        
        json_data = df.to_json(orient='records', indent=2)
        b64 = base64.b64encode(json_data.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="fund_data.json" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #ab47bc, #9c27b0); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; width: 100%; margin-top: 10px;">🔧 Download JSON</button></a>'
        st.markdown(href, unsafe_allow_html=True)

# Enhanced main application with professional styling
def main():
    model = initialize_gemini()
    cache = get_context_cache()
    
    # Enhanced sidebar with professional styling
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        
        # Enhanced cache information
        st.markdown("### 🧠 Smart Caching")
        # st.info("💡 **AI Context Caching**: Reduces token usage and improves response times by caching analysis results.")
        
        # Cache management with enhanced styling
        if st.button("🗑️ Clear Analysis Cache", use_container_width=True):
            cache.cache_store.clear()
            st.success("✅ Cache cleared successfully!")
        
        # Enhanced cache stats
        st.markdown("### 📊 Cache Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cached Files", len(cache.cache_store))
        with col2:
            cache_size = sum(len(str(data)) for data, _ in cache.cache_store.values())
            st.metric("Cache Size", f"{cache_size//1024}KB")
        
        # Additional information
        st.markdown("### 📋 Supported Formats")
        st.markdown("""
        - **PDF** documents
        - **Excel** spreadsheets (.xlsx, .xls)
        - **CSV** files
        - **Text** files
        """)
        
        st.markdown("### 🎯 What We Extract")
        st.markdown("""
        - Fund names & returns
        - Assets Under Management
        - Investment strategies
        - Performance metrics
        """)
    
    # Simple file uploader section
    st.markdown("## 📁 Upload Fund Documents")
    st.markdown("Upload multiple files in any supported format. Intelligently extract and analyze fund performance data.")
    
    uploaded_files = st.file_uploader(
        "Choose files to analyze", 
        accept_multiple_files=True,
        type=['pdf', 'xlsx', 'xls', 'csv', 'txt'],
        help="💡 Supported formats: PDF, Excel, CSV, Text files"
    )
    
    if uploaded_files:
        # Processing section
        st.markdown(f"## 🔄 Processing {len(uploaded_files)} Files")
        
        all_dataframes = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            file_content = uploaded_file.read()
            filename = uploaded_file.name
            
            text_content = extract_text_content_enhanced(file_content, filename)
            
            if text_content.strip():
                df = extract_fund_data_with_gemini_enhanced(model, text_content, filename, cache)
                
                if not df.empty:
                    df = standardize_dataframe(df)
                    df['source_file'] = filename
                    all_dataframes.append(df)
                else:
                    st.warning(f"⚠️ No fund data extracted from {filename}")
            else:
                st.error(f"❌ Could not extract text from {filename}")
        
        progress_bar.empty()
        status_text.empty()
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Success message
            st.success(f"🎉 Successfully processed {len(all_dataframes)} files and extracted {len(combined_df)} fund records!")
            
            # Data display
            st.markdown("## 📋 Extracted Fund Data")
            st.dataframe(combined_df, use_container_width=True, height=400)
            
            # Key metrics
            st.markdown("## 📊 Portfolio Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Funds", len(combined_df))
            
            with col2:
                if 'strategy' in combined_df.columns:
                    strategies = combined_df['strategy'].nunique()
                    st.metric("Strategies", strategies)
            
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
            
            # Troubleshooting section
            st.markdown("## 🔧 Troubleshooting Tips")
            st.warning("""
            **💡 Tips for Better Results:**
            
            - **Document Quality:** Ensure files contain actual fund performance data
            - **PDF Issues:** Check that text is not image-based (try copying text from the PDF)
            - **Data Format:** Verify that data is in a readable table or structured format
            - **File Size:** Very large files may have extraction issues
            - **Content Type:** Marketing materials may have limited extractable data
            """)
    
    else:
        # Welcome section using Streamlit native components
        st.markdown("## 🚀 Welcome to Fund Analysis")
        
        # Feature cards using columns
        col1, col2 = st.columns(2)
        
        # with col1:
        #     st.markdown("### 📁 Upload & Process")
        #     st.info("""
        #     **Multi-format Support**
            
        #     Support for PDF, Excel, CSV, and text files. Dashboard handles diverse document formats and complex data structures.
            
        #     ✅ Multi-format document support  
        #     ✅ Intelligent text extraction  
        #     ✅ Error handling & fallbacks
        #     """)
            
        #     st.markdown("### 📊 Professional Insights")
        #     st.success("""
        #     **Advanced Analytics**
            
        #     Generate comprehensive analysis with interactive visualizations and detailed performance metrics.
            
        #     ✅ Interactive dashboards  
        #     ✅ Risk analysis & metrics  
        #     ✅ Strategy comparisons
        #     """)
        
        # with col2:
        #     st.markdown("### Analysis")
        #     st.info("""
        #     **Expert-Level Intelligence**
            
        #     Extracts fund data.
            
        #     ✅ Intelligent data extraction  
        #     ✅ Strategy classification  
        #     ✅ Performance standardization
        #     """)
            
        #     st.markdown("### ⬇️ Export Results")
        #     st.success("""
        #     **Multiple Formats**
            
        #     Download comprehensive reports in multiple formats for presentations and further analysis.
            
        #     ✅ Excel with multiple sheets  
        #     ✅ CSV for data analysis  
        #     ✅ JSON for developers
        #     """)
        
        # What we extract section
        st.markdown("## 🎯 What Our Dashboard Can Extract")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 📈 Performance Data")
            st.markdown("""
            - Weekly, monthly, quarterly returns
            - YTD and since inception performance
            - Absolute and percentage returns
            """)
        
        with col2:
            st.markdown("### 💰 Fund Information")
            st.markdown("""
            - Assets Under Management (AUM)
            - Net Asset Value (NAV)
            - Fund names and identifiers
            """)
        
        with col3:
            st.markdown("### 🎯 Investment Strategies")
            st.markdown("""
            - Long/Short Equity, Credit, Macro
            - Event-Driven, Quantitative
            - Multi-Strategy classifications
            """)
        
        with col4:
            st.markdown("### 📊 Additional Metrics")
            st.markdown("""
            - Risk metrics and volatility
            - Sharpe ratios and drawdowns
            - Benchmark comparisons
            """)
        
        # Call to action
        st.markdown("---")
        st.markdown("### 🎯 Ready to Get Started?")
        st.info("""
        **Upload your fund documents above.**
        """)

if __name__ == "__main__":
    main()
