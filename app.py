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

# Enhanced file processing functions (add these as separate functions, not part of the class)

def extract_text_content_enhanced(file_content: bytes, filename: str) -> str:
    """Enhanced text extraction with multiple fallback methods"""
    filename_lower = filename.lower()
    
    text_content = ""
    
    if filename_lower.endswith('.pdf'):
        # Try multiple PDF extraction methods
        text_content = extract_pdf_with_fallbacks(file_content)
    elif filename_lower.endswith(('.xlsx', '.xls')):
        text_content = extract_excel_enhanced(file_content, filename)
    elif filename_lower.endswith('.csv'):
        text_content = extract_csv_enhanced(file_content)
    elif filename_lower.endswith('.txt'):
        text_content = file_content.decode('utf-8', errors='ignore')
    else:
        # Try to decode as text for unknown formats
        try:
            text_content = file_content.decode('utf-8', errors='ignore')
        except:
            st.warning(f"Unsupported file format: {filename}")
            return ""
    
    # If extraction failed or produced minimal content, try alternate methods
    if len(text_content.strip()) < 50:
        st.warning(f"Limited text extracted from {filename}. Trying alternate methods...")
        # Could add OCR or other fallback methods here
    
    return text_content

def extract_pdf_with_fallbacks(file_content: bytes) -> str:
    """Enhanced PDF extraction with multiple methods"""
    text = ""
    
    try:
        # Method 1: PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        if len(text.strip()) < 50:
            st.info("PyPDF2 extraction yielded minimal text. Document may be image-based.")
            # Here you could add other PDF libraries like pdfplumber, pymupdf, etc.
            
    except Exception as e:
        st.warning(f"PDF extraction failed with PyPDF2: {e}")
        # Could try other PDF extraction libraries here
    
    return text

def extract_excel_enhanced(file_content: bytes, filename: str) -> str:
    """Enhanced Excel extraction handling complex structures"""
    try:
        # Read all sheets and handle different structures
        excel_data = pd.read_excel(BytesIO(file_content), sheet_name=None, header=None)
        text_content = []
        
        for sheet_name, df in excel_data.items():
            text_content.append(f"=== Sheet: {sheet_name} ===")
            
            # Try to detect header rows and data structure
            # Look for common fund report patterns
            for i, row in df.iterrows():
                row_text = " | ".join([str(cell) for cell in row.dropna()])
                if len(row_text.strip()) > 0:
                    text_content.append(row_text)
            
            text_content.append("\n")
        
        return "\n".join(text_content)
    except Exception as e:
        st.warning(f"Enhanced Excel extraction failed: {e}")
        # Fallback to basic extraction
        try:
            df = pd.read_excel(BytesIO(file_content))
            return df.to_string(index=False)
        except:
            return ""

def extract_csv_enhanced(file_content: bytes) -> str:
    """Enhanced CSV extraction with encoding detection"""
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
            try:
                text = file_content.decode(encoding)
                df = pd.read_csv(BytesIO(file_content), encoding=encoding)
                return df.to_string(index=False)
            except:
                continue
        
        # If all encodings fail, try with error handling
        df = pd.read_csv(BytesIO(file_content), encoding='utf-8', errors='ignore')
        return df.to_string(index=False)
        
    except Exception as e:
        st.warning(f"CSV extraction failed: {e}")
        return ""

def extract_fund_data_with_gemini_enhanced(model, text_content: str, filename: str, cache: ContextCache) -> pd.DataFrame:
    """Enhanced Gemini extraction with better error handling and validation"""
    
    # Check cache first
    cached_result = cache.get_cached_context(text_content)
    if cached_result is not None:
        st.info(f"Using cached analysis for {filename}")
        return cached_result
    
    try:
        system_context = cache.create_robust_system_context()
        
        # Enhanced prompt with more context about the specific document
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
        
        with st.spinner(f"AI analyzing {filename} with enhanced intelligence..."):
            response = model.generate_content(prompt)
            
            # Enhanced JSON parsing with better error handling
            response_text = response.text.strip()
            
            # Multiple attempts to extract JSON
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
                # Remove common AI response prefixes
                cleaned = response_text.replace("Here's the extracted data:", "")
                cleaned = cleaned.replace("Based on the document:", "")
                # Look for JSON-like structures
                # import re
                json_pattern = r'\[.*?\]'
                matches = re.findall(json_pattern, cleaned, re.DOTALL)
                if matches:
                    json_text = matches[0]
            
            if json_text is None:
                st.warning(f"Could not find valid JSON in AI response for {filename}")
                st.write("Raw AI Response:", response_text[:500])
                return pd.DataFrame()
            
            # Parse JSON with error handling
            try:
                fund_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                st.warning(f"JSON parsing failed for {filename}. Attempting to fix...")
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
                except:
                    st.error(f"Could not parse JSON for {filename}")
                    st.write("Cleaned JSON:", json_text[:500])
                    return pd.DataFrame()
            
            df = pd.DataFrame(fund_data)
            
            # Enhanced data validation and cleaning
            if len(df) > 0:
                # Validate and clean the data
                df = validate_and_clean_extracted_data(df, filename)
            
            # Cache the result
            cache.set_cached_context(text_content, df)
            
            confidence_info = ""
            if 'confidence' in df.columns:
                high_conf = (df['confidence'] == 'High').sum()
                med_conf = (df['confidence'] == 'Medium').sum()
                low_conf = (df['confidence'] == 'Low').sum()
                confidence_info = f" (Confidence: {high_conf}H/{med_conf}M/{low_conf}L)"
            
            st.success(f"✅ Extracted {len(df)} funds from {filename}{confidence_info}")
            return df
            
    except Exception as e:
        st.error(f"Enhanced AI extraction failed for {filename}: {e}")
        st.write("This might be due to:")
        st.write("- Document is image-based and needs OCR")
        st.write("- Document has unusual structure")
        st.write("- API rate limits or connection issues")
        return pd.DataFrame()

def validate_and_clean_extracted_data(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Validate and clean extracted fund data"""
    
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
    df = df[df['fund_name'] != 'nan']  # Remove rows with no fund name
    df = df[df['fund_name'].str.len() > 2]  # Remove very short names
    
    # Clean and validate returns
    if 'return' in df.columns:
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        # Flag suspicious returns
        suspicious_returns = df[(df['return'].abs() > 2.0) & df['return'].notna()]
        if len(suspicious_returns) > 0:
            st.warning(f"Found {len(suspicious_returns)} funds with returns > 200% in {filename}")
    
    # Clean and validate AUM
    if 'aum' in df.columns:
        df['aum'] = pd.to_numeric(df['aum'], errors='coerce')
        # Flag suspicious AUM
        suspicious_aum = df[(df['aum'] > 500000) & df['aum'].notna()]  # > $500B
        if len(suspicious_aum) > 0:
            st.warning(f"Found {len(suspicious_aum)} funds with AUM > $500B in {filename}")
    
    # Clean strategies
    if 'strategy' in df.columns:
        df['strategy'] = df['strategy'].astype(str).str.strip()
        df.loc[df['strategy'].isin(['None', 'nan', 'null', '']) | df['strategy'].isnull(), 'strategy'] = 'Multi-Strategy'
    
    # Remove duplicate fund names (keep first occurrence)
    df = df.drop_duplicates(subset=['fund_name'], keep='first')
    
    final_count = len(df)
    if final_count < original_count:
        st.info(f"Cleaned data: {original_count} → {final_count} funds in {filename}")
    
    return df

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
# Part 2: Visualization Functions and Main Application

def create_visualizations(df: pd.DataFrame):
    """Create intuitive and meaningful fund analysis visualizations"""
    if df.empty:
        st.warning("No data available for visualization")
        return
    
    st.subheader("📊 Fund Performance Dashboard")
    
    # Create more intuitive tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Performance Ranking", "💰 Capital Analysis", "📈 Strategy Insights", "⚠️ Risk Assessment"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.write("**🏆 Fund Performance Leaderboard**")
                
                # Create a more intuitive performance chart
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Sort funds by performance
                sorted_df = df.sort_values('return', ascending=True)
                fund_names = [name[:20] + "..." if len(name) > 20 else name for name in sorted_df['fund_name']]
                returns = sorted_df['return'].values
                
                # Color code: Green for positive, Red for negative, Yellow for near-zero
                colors = []
                for ret in returns:
                    if ret > 0.005:  # > 0.5%
                        colors.append('#2E8B57')  # Strong Green
                    elif ret > 0:
                        colors.append('#90EE90')  # Light Green
                    elif ret > -0.005:  # > -0.5%
                        colors.append('#FFD700')  # Yellow
                    else:
                        colors.append('#DC143C')  # Red
                
                bars = ax.barh(range(len(returns)), returns, color=colors)
                ax.set_yticks(range(len(returns)))
                ax.set_yticklabels(fund_names, fontsize=10)
                ax.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
                ax.set_title('Fund Performance Ranking', fontsize=14, fontweight='bold', pad=20)
                
                # Add value labels
                for i, v in enumerate(returns):
                    label_color = 'white' if abs(v) > 0.005 else 'black'
                    ax.text(v/2, i, f'{v:.2%}', va='center', ha='center', 
                           fontweight='bold', color=label_color, fontsize=9)
                
                # Add zero line
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'return' in df.columns and 'strategy' in df.columns:
                st.write("**📊 Performance by Strategy (Box Plot)**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create strategy performance box plot
                strategy_data = []
                strategy_labels = []
                
                for strategy in df['strategy'].unique():
                    if pd.notna(strategy):
                        strategy_returns = df[df['strategy'] == strategy]['return'].dropna()
                        if len(strategy_returns) > 0:
                            strategy_data.append(strategy_returns.values)
                            strategy_labels.append(strategy[:15])
                
                if strategy_data:
                    bp = ax.boxplot(strategy_data, labels=strategy_labels, patch_artist=True)
                    
                    # Color the boxes
                    colors = plt.cm.Set3(range(len(bp['boxes'])))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
                    ax.set_title('Return Distribution by Strategy', fontsize=14, fontweight='bold', pad=20)
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(axis='y', alpha=0.3)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'aum' in df.columns and not df['aum'].isnull().all():
                st.write("**💰 Capital Allocation Overview**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create a treemap-style visualization using a pie chart with better labels
                strategy_aum = df.groupby('strategy')['aum'].sum().sort_values(ascending=False)
                
                # Custom colors for better visualization
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FCEA2B', '#FF9F43', '#EE5A24']
                
                wedges, texts, autotexts = ax.pie(strategy_aum.values, 
                                                 labels=strategy_aum.index,
                                                 autopct=lambda pct: f'${strategy_aum.sum()*pct/100:.0f}M\n({pct:.1f}%)',
                                                 colors=colors[:len(strategy_aum)],
                                                 startangle=90,
                                                 textprops={'fontsize': 10})
                
                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                
                ax.set_title('AUM Distribution by Strategy', fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'aum' in df.columns and 'return' in df.columns:
                st.write("**📈 Capital vs Performance Analysis**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Scatter plot of AUM vs Return with fund labels
                scatter = ax.scatter(df['return'], df['aum'], 
                                   s=100, alpha=0.7, 
                                   c=range(len(df)), cmap='viridis')
                
                # Add fund labels
                for i, row in df.iterrows():
                    if pd.notna(row['return']) and pd.notna(row['aum']):
                        label = row['fund_name'][:15] + "..." if len(row['fund_name']) > 15 else row['fund_name']
                        ax.annotate(label, (row['return'], row['aum']), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
                
                ax.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
                ax.set_ylabel('AUM ($M)', fontsize=12, fontweight='bold')
                ax.set_title('Fund Size vs Performance', fontsize=14, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'strategy' in df.columns and 'return' in df.columns:
                st.write("**🎯 Strategy Performance Comparison**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Calculate strategy metrics
                strategy_stats = df.groupby('strategy').agg({
                    'return': ['mean', 'count'],
                    'aum': 'sum'
                }).round(4)
                
                strategy_stats.columns = ['Avg_Return', 'Fund_Count', 'Total_AUM']
                strategy_stats = strategy_stats.sort_values('Avg_Return', ascending=True)
                
                # Create horizontal bar chart with fund count as bar width
                colors = ['red' if x < 0 else 'lightgreen' if x < 0.005 else 'green' 
                         for x in strategy_stats['Avg_Return']]
                
                bars = ax.barh(range(len(strategy_stats)), strategy_stats['Avg_Return'], 
                              color=colors, alpha=0.8)
                
                ax.set_yticks(range(len(strategy_stats)))
                ax.set_yticklabels(strategy_stats.index, fontsize=10)
                ax.set_xlabel('Average Return (%)', fontsize=12, fontweight='bold')
                ax.set_title('Strategy Performance Ranking', fontsize=14, fontweight='bold', pad=20)
                
                # Add value and fund count labels
                for i, (ret, count) in enumerate(zip(strategy_stats['Avg_Return'], strategy_stats['Fund_Count'])):
                    ax.text(ret + 0.001 if ret > 0 else ret - 0.001, i, 
                           f'{ret:.2%} ({int(count)} funds)', 
                           va='center', ha='left' if ret > 0 else 'right', fontsize=9)
                
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'strategy' in df.columns:
                st.write("**📋 Strategy Summary Table**")
                
                # Create comprehensive strategy summary
                strategy_summary = df.groupby('strategy').agg({
                    'return': ['mean', 'std', 'min', 'max', 'count'],
                    'aum': ['sum', 'mean']
                }).round(4)
                
                # Flatten column names
                strategy_summary.columns = ['Avg Return', 'Volatility', 'Min Return', 'Max Return', 
                                          'Funds', 'Total AUM', 'Avg Fund Size']
                
                # Format for display
                display_summary = strategy_summary.copy()
                for col in ['Avg Return', 'Volatility', 'Min Return', 'Max Return']:
                    display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                
                for col in ['Total AUM', 'Avg Fund Size']:
                    display_summary[col] = display_summary[col].apply(lambda x: f"${x:.0f}M" if pd.notna(x) else "N/A")
                
                display_summary['Funds'] = display_summary['Funds'].astype(int)
                
                st.dataframe(display_summary, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.write("**⚠️ Risk Distribution Analysis**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create risk buckets
                returns_clean = df['return'].dropna()
                
                # Define risk categories
                high_risk = returns_clean[abs(returns_clean) > 0.01].count()
                medium_risk = returns_clean[(abs(returns_clean) > 0.005) & (abs(returns_clean) <= 0.01)].count()
                low_risk = returns_clean[abs(returns_clean) <= 0.005].count()
                
                categories = ['Low Risk\n(±0.5%)', 'Medium Risk\n(±0.5-1%)', 'High Risk\n(>±1%)']
                counts = [low_risk, medium_risk, high_risk]
                colors = ['green', 'orange', 'red']
                
                bars = ax.bar(categories, counts, color=colors, alpha=0.7)
                ax.set_ylabel('Number of Funds', fontsize=12, fontweight='bold')
                ax.set_title('Risk Profile Distribution', fontsize=14, fontweight='bold', pad=20)
                
                # Add count labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold')
                
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.write("**📊 Portfolio Risk Metrics**")
                
                returns_clean = df['return'].dropna()
                
                if len(returns_clean) > 0:
                    # Calculate comprehensive risk metrics
                    portfolio_return = returns_clean.mean()
                    portfolio_vol = returns_clean.std()
                    downside_deviation = returns_clean[returns_clean < 0].std() if len(returns_clean[returns_clean < 0]) > 0 else 0
                    max_drawdown = returns_clean.min()
                    upside_capture = returns_clean[returns_clean > 0].mean() if len(returns_clean[returns_clean > 0]) > 0 else 0
                    win_rate = (returns_clean > 0).sum() / len(returns_clean)
                    
                    # Create metrics display
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
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

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
            text_content = extract_text_content_enhanced(file_content, filename)
            
            if text_content.strip():
                # Use Gemini to extract fund data
                df = extract_fund_data_with_gemini_enhanced(model, text_content, filename, cache)
                
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
    main()# Part 2: Visualization Functions and Main Application

def create_visualizations(df: pd.DataFrame):
    """Create intuitive and meaningful fund analysis visualizations"""
    if df.empty:
        st.warning("No data available for visualization")
        return
    
    st.subheader("📊 Fund Performance Dashboard")
    
    # Create more intuitive tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Performance Ranking", "💰 Capital Analysis", "📈 Strategy Insights", "⚠️ Risk Assessment"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.write("**🏆 Fund Performance Leaderboard**")
                
                # Create a more intuitive performance chart
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Sort funds by performance
                sorted_df = df.sort_values('return', ascending=True)
                fund_names = [name[:20] + "..." if len(name) > 20 else name for name in sorted_df['fund_name']]
                returns = sorted_df['return'].values
                
                # Color code: Green for positive, Red for negative, Yellow for near-zero
                colors = []
                for ret in returns:
                    if ret > 0.005:  # > 0.5%
                        colors.append('#2E8B57')  # Strong Green
                    elif ret > 0:
                        colors.append('#90EE90')  # Light Green
                    elif ret > -0.005:  # > -0.5%
                        colors.append('#FFD700')  # Yellow
                    else:
                        colors.append('#DC143C')  # Red
                
                bars = ax.barh(range(len(returns)), returns, color=colors)
                ax.set_yticks(range(len(returns)))
                ax.set_yticklabels(fund_names, fontsize=10)
                ax.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
                ax.set_title('Fund Performance Ranking', fontsize=14, fontweight='bold', pad=20)
                
                # Add value labels
                for i, v in enumerate(returns):
                    label_color = 'white' if abs(v) > 0.005 else 'black'
                    ax.text(v/2, i, f'{v:.2%}', va='center', ha='center', 
                           fontweight='bold', color=label_color, fontsize=9)
                
                # Add zero line
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'return' in df.columns and 'strategy' in df.columns:
                st.write("**📊 Performance by Strategy (Box Plot)**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create strategy performance box plot
                strategy_data = []
                strategy_labels = []
                
                for strategy in df['strategy'].unique():
                    if pd.notna(strategy):
                        strategy_returns = df[df['strategy'] == strategy]['return'].dropna()
                        if len(strategy_returns) > 0:
                            strategy_data.append(strategy_returns.values)
                            strategy_labels.append(strategy[:15])
                
                if strategy_data:
                    bp = ax.boxplot(strategy_data, labels=strategy_labels, patch_artist=True)
                    
                    # Color the boxes
                    colors = plt.cm.Set3(range(len(bp['boxes'])))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
                    ax.set_title('Return Distribution by Strategy', fontsize=14, fontweight='bold', pad=20)
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(axis='y', alpha=0.3)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'aum' in df.columns and not df['aum'].isnull().all():
                st.write("**💰 Capital Allocation Overview**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create a treemap-style visualization using a pie chart with better labels
                strategy_aum = df.groupby('strategy')['aum'].sum().sort_values(ascending=False)
                
                # Custom colors for better visualization
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FCEA2B', '#FF9F43', '#EE5A24']
                
                wedges, texts, autotexts = ax.pie(strategy_aum.values, 
                                                 labels=strategy_aum.index,
                                                 autopct=lambda pct: f'${strategy_aum.sum()*pct/100:.0f}M\n({pct:.1f}%)',
                                                 colors=colors[:len(strategy_aum)],
                                                 startangle=90,
                                                 textprops={'fontsize': 10})
                
                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                
                ax.set_title('AUM Distribution by Strategy', fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'aum' in df.columns and 'return' in df.columns:
                st.write("**📈 Capital vs Performance Analysis**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Scatter plot of AUM vs Return with fund labels
                scatter = ax.scatter(df['return'], df['aum'], 
                                   s=100, alpha=0.7, 
                                   c=range(len(df)), cmap='viridis')
                
                # Add fund labels
                for i, row in df.iterrows():
                    if pd.notna(row['return']) and pd.notna(row['aum']):
                        label = row['fund_name'][:15] + "..." if len(row['fund_name']) > 15 else row['fund_name']
                        ax.annotate(label, (row['return'], row['aum']), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
                
                ax.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
                ax.set_ylabel('AUM ($M)', fontsize=12, fontweight='bold')
                ax.set_title('Fund Size vs Performance', fontsize=14, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'strategy' in df.columns and 'return' in df.columns:
                st.write("**🎯 Strategy Performance Comparison**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Calculate strategy metrics
                strategy_stats = df.groupby('strategy').agg({
                    'return': ['mean', 'count'],
                    'aum': 'sum'
                }).round(4)
                
                strategy_stats.columns = ['Avg_Return', 'Fund_Count', 'Total_AUM']
                strategy_stats = strategy_stats.sort_values('Avg_Return', ascending=True)
                
                # Create horizontal bar chart with fund count as bar width
                colors = ['red' if x < 0 else 'lightgreen' if x < 0.005 else 'green' 
                         for x in strategy_stats['Avg_Return']]
                
                bars = ax.barh(range(len(strategy_stats)), strategy_stats['Avg_Return'], 
                              color=colors, alpha=0.8)
                
                ax.set_yticks(range(len(strategy_stats)))
                ax.set_yticklabels(strategy_stats.index, fontsize=10)
                ax.set_xlabel('Average Return (%)', fontsize=12, fontweight='bold')
                ax.set_title('Strategy Performance Ranking', fontsize=14, fontweight='bold', pad=20)
                
                # Add value and fund count labels
                for i, (ret, count) in enumerate(zip(strategy_stats['Avg_Return'], strategy_stats['Fund_Count'])):
                    ax.text(ret + 0.001 if ret > 0 else ret - 0.001, i, 
                           f'{ret:.2%} ({int(count)} funds)', 
                           va='center', ha='left' if ret > 0 else 'right', fontsize=9)
                
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'strategy' in df.columns:
                st.write("**📋 Strategy Summary Table**")
                
                # Create comprehensive strategy summary
                strategy_summary = df.groupby('strategy').agg({
                    'return': ['mean', 'std', 'min', 'max', 'count'],
                    'aum': ['sum', 'mean']
                }).round(4)
                
                # Flatten column names
                strategy_summary.columns = ['Avg Return', 'Volatility', 'Min Return', 'Max Return', 
                                          'Funds', 'Total AUM', 'Avg Fund Size']
                
                # Format for display
                display_summary = strategy_summary.copy()
                for col in ['Avg Return', 'Volatility', 'Min Return', 'Max Return']:
                    display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                
                for col in ['Total AUM', 'Avg Fund Size']:
                    display_summary[col] = display_summary[col].apply(lambda x: f"${x:.0f}M" if pd.notna(x) else "N/A")
                
                display_summary['Funds'] = display_summary['Funds'].astype(int)
                
                st.dataframe(display_summary, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.write("**⚠️ Risk Distribution Analysis**")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create risk buckets
                returns_clean = df['return'].dropna()
                
                # Define risk categories
                high_risk = returns_clean[abs(returns_clean) > 0.01].count()
                medium_risk = returns_clean[(abs(returns_clean) > 0.005) & (abs(returns_clean) <= 0.01)].count()
                low_risk = returns_clean[abs(returns_clean) <= 0.005].count()
                
                categories = ['Low Risk\n(±0.5%)', 'Medium Risk\n(±0.5-1%)', 'High Risk\n(>±1%)']
                counts = [low_risk, medium_risk, high_risk]
                colors = ['green', 'orange', 'red']
                
                bars = ax.bar(categories, counts, color=colors, alpha=0.7)
                ax.set_ylabel('Number of Funds', fontsize=12, fontweight='bold')
                ax.set_title('Risk Profile Distribution', fontsize=14, fontweight='bold', pad=20)
                
                # Add count labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold')
                
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            if 'return' in df.columns and not df['return'].isnull().all():
                st.write("**📊 Portfolio Risk Metrics**")
                
                returns_clean = df['return'].dropna()
                
                if len(returns_clean) > 0:
                    # Calculate comprehensive risk metrics
                    portfolio_return = returns_clean.mean()
                    portfolio_vol = returns_clean.std()
                    downside_deviation = returns_clean[returns_clean < 0].std() if len(returns_clean[returns_clean < 0]) > 0 else 0
                    max_drawdown = returns_clean.min()
                    upside_capture = returns_clean[returns_clean > 0].mean() if len(returns_clean[returns_clean > 0]) > 0 else 0
                    win_rate = (returns_clean > 0).sum() / len(returns_clean)
                    
                    # Create metrics display
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
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

def generate_executive_summary_pdf(df: pd.DataFrame) -> bytes:
    """Generate a clean, professional executive summary PDF"""
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime
    
    # Create PDF
    pdf_buffer = BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Create figure with clean layout
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
        fig.patch.set_facecolor('white')
        
        # === HEADER SECTION ===
        fig.text(0.5, 0.95, 'FUND PERFORMANCE EXECUTIVE SUMMARY', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Header details
        report_date = datetime.now().strftime('%B %d, %Y')
        total_funds = len(df)
        total_strategies = df["strategy"].nunique() if "strategy" in df.columns else 0
        
        fig.text(0.1, 0.91, f'Report Date: {report_date}', fontsize=10)
        fig.text(0.5, 0.91, f'Portfolio: {total_funds} Funds | {total_strategies} Strategies', 
                fontsize=10)
