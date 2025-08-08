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
        
        SPECIFIC EXAMPLES from your training:
        - "Crest Quant Alpha": 1.45% -> strategy should be "Quantitative" (from "Quant")
        - "Crest Merger Fund" -> strategy should be "Event-Driven" (merger arbitrage)
        - "Boreal Credit Opps" -> strategy should be "Credit" (from fund name)
        - "Atlas Select" with "Long/Short Eq" -> "Long/Short Equity"
        - "Atlas Currency" with "Global Macro" -> "Global Macro"
        
        Please extract all fund performance data from this document and return as valid JSON array.
        Focus on accuracy and completeness. Apply senior analyst expertise to interpret the data correctly.
        Ensure EVERY fund has a strategy assigned - never leave strategy as null or empty.
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
                fontsize=10, ha='center')
        
        if 'aum' in df.columns and not df['aum'].isnull().all():
            total_aum = df['aum'].sum()
            fig.text(0.9, 0.91, f'Total AUM: ${total_aum:,.0f}M', fontsize=10, ha='right')
        
        # Separator line
        fig.add_artist(plt.Line2D([0.1, 0.9], [0.89, 0.89], color='black', linewidth=0.5))
        
        # === KEY METRICS CARDS ===
        if 'return' in df.columns and not df['return'].isnull().all():
            avg_return = df['return'].mean()
            best_return = df['return'].max()
            win_rate = (df['return'] > 0).sum() / len(df[df['return'].notna()])
            volatility = df['return'].std()
            
            # Create metric cards
            y_pos = 0.82
            
            # Card 1: Portfolio Return
            fig.text(0.2, y_pos, 'PORTFOLIO RETURN', fontsize=12, fontweight='bold', ha='center')
            color = 'green' if avg_return > 0 else 'red'
            fig.text(0.2, y_pos - 0.04, f'{avg_return:.2%}', fontsize=20, fontweight='bold', 
                    ha='center', color=color)
            
            # Card 2: Best Performer
            fig.text(0.5, y_pos, 'BEST PERFORMER', fontsize=12, fontweight='bold', ha='center')
            fig.text(0.5, y_pos - 0.04, f'{best_return:.2%}', fontsize=20, fontweight='bold', 
                    ha='center', color='green')
            
            # Card 3: Win Rate
            fig.text(0.8, y_pos, 'WIN RATE', fontsize=12, fontweight='bold', ha='center')
            color = 'green' if win_rate > 0.5 else 'red'
            fig.text(0.8, y_pos - 0.04, f'{win_rate:.1%}', fontsize=20, fontweight='bold', 
                    ha='center', color=color)
        
        # === TOP PERFORMERS TABLE ===
        if 'return' in df.columns and not df['return'].isnull().all():
            # Position for table
            table_y = 0.72
            fig.text(0.1, table_y, 'TOP PERFORMERS', fontsize=14, fontweight='bold')
            
            # Get top 5 performers
            top_performers = df.nlargest(5, 'return')[['fund_name', 'return', 'strategy']]
            
            # Table headers
            fig.text(0.1, table_y - 0.03, 'Fund Name', fontsize=10, fontweight='bold')
            fig.text(0.5, table_y - 0.03, 'Return', fontsize=10, fontweight='bold')
            fig.text(0.7, table_y - 0.03, 'Strategy', fontsize=10, fontweight='bold')
            
            # Table data
            for i, (_, row) in enumerate(top_performers.iterrows()):
                y = table_y - 0.06 - (i * 0.025)
                fund_name = row['fund_name'][:25] + "..." if len(row['fund_name']) > 25 else row['fund_name']
                strategy = str(row['strategy'])[:15] + "..." if len(str(row['strategy'])) > 15 else str(row['strategy'])
                
                fig.text(0.1, y, fund_name, fontsize=9)
                fig.text(0.5, y, f"{row['return']:.2%}", fontsize=9, 
                        color='green' if row['return'] > 0 else 'red')
                fig.text(0.7, y, strategy, fontsize=9)
        
        # === CHART 1: Performance Ranking ===
        ax1 = plt.axes([0.1, 0.35, 0.35, 0.25])  # [left, bottom, width, height]
        
        if 'return' in df.columns and not df['return'].isnull().all():
            # Sort and limit to top/bottom performers
            sorted_df = df.sort_values('return', ascending=False)
            display_df = pd.concat([sorted_df.head(3), sorted_df.tail(2)])
            
            fund_names = [name[:12] + "..." if len(name) > 12 else name for name in display_df['fund_name']]
            returns = display_df['return'].values
            
            colors = ['green' if r > 0 else 'red' for r in returns]
            
            bars = ax1.barh(range(len(returns)), returns, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(returns)))
            ax1.set_yticklabels(fund_names, fontsize=8)
            ax1.set_xlabel('Return', fontsize=9)
            ax1.set_title('Fund Performance', fontsize=11, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for i, v in enumerate(returns):
                ax1.text(v + (0.001 if v > 0 else -0.001), i, f'{v:.1%}', 
                        va='center', ha='left' if v > 0 else 'right', fontsize=8)
        
        # === CHART 2: Strategy Performance ===
        ax2 = plt.axes([0.55, 0.35, 0.35, 0.25])
        
        if 'strategy' in df.columns and 'return' in df.columns:
            strategy_returns = df.groupby('strategy')['return'].mean().sort_values(ascending=True)
            
            colors = ['red' if r < 0 else 'lightgreen' if r < 0.005 else 'green' 
                     for r in strategy_returns.values]
            
            bars = ax2.barh(range(len(strategy_returns)), strategy_returns.values, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(strategy_returns)))
            ax2.set_yticklabels([s[:12] + "..." if len(s) > 12 else s for s in strategy_returns.index], fontsize=8)
            ax2.set_xlabel('Avg Return', fontsize=9)
            ax2.set_title('Strategy Performance', fontsize=11, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for i, v in enumerate(strategy_returns.values):
                ax2.text(v + (0.0005 if v > 0 else -0.0005), i, f'{v:.1%}', 
                        va='center', ha='left' if v > 0 else 'right', fontsize=8)
        
        # === AUM ALLOCATION PIE CHART ===
        if 'aum' in df.columns and 'strategy' in df.columns and not df['aum'].isnull().all():
            ax3 = plt.axes([0.1, 0.05, 0.35, 0.25])
            
            strategy_aum = df.groupby('strategy')['aum'].sum().sort_values(ascending=False)
            
            # Limit to top 5 strategies
            if len(strategy_aum) > 5:
                top_strategies = strategy_aum.head(5)
                other_aum = strategy_aum.tail(-5).sum()
                if other_aum > 0:
                    top_strategies['Other'] = other_aum
                strategy_aum = top_strategies
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FCEA2B', '#FF9F43']
            
            wedges, texts, autotexts = ax3.pie(strategy_aum.values, 
                                              labels=[s[:10] + "..." if len(s) > 10 else s for s in strategy_aum.index],
                                              autopct='%1.1f%%',
                                              colors=colors[:len(strategy_aum)],
                                              startangle=90)
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            
            for text in texts:
                text.set_fontsize(8)
            
            ax3.set_title('AUM Allocation', fontsize=11, fontweight='bold')
        
        # === KEY INSIGHTS TABLE ===
        ax4 = plt.axes([0.55, 0.05, 0.35, 0.25])
        ax4.axis('off')
        
        # Create insights summary
        insights = []
        
        if 'return' in df.columns and not df['return'].isnull().all():
            volatility = df['return'].std()
            positive_funds = (df['return'] > 0).sum()
            total_funds_with_return = len(df[df['return'].notna()])
            
            insights.extend([
                f"Portfolio Volatility: {volatility:.2%}",
                f"Positive Returns: {positive_funds}/{total_funds_with_return}",
                f"Risk Level: {'Low' if volatility < 0.005 else 'Medium' if volatility < 0.015 else 'High'}"
            ])
        
        if 'strategy' in df.columns and 'return' in df.columns:
            best_strategy = df.groupby('strategy')['return'].mean().idxmax()
            strategy_count = df['strategy'].nunique()
            insights.extend([
                f"Best Strategy: {best_strategy[:15]}",
                f"Total Strategies: {strategy_count}"
            ])
        
        if 'aum' in df.columns and not df['aum'].isnull().all():
            largest_fund = df.loc[df['aum'].idxmax(), 'fund_name']
            avg_fund_size = df['aum'].mean()
            insights.extend([
                f"Largest Fund: {largest_fund[:15]}",
                f"Avg Fund Size: ${avg_fund_size:.0f}M"
            ])
        
        # Display insights
        ax4.text(0.5, 0.9, 'KEY INSIGHTS', fontsize=11, fontweight='bold', ha='center')
        
        for i, insight in enumerate(insights):
            ax4.text(0.05, 0.75 - i*0.12, f"• {insight}", fontsize=9, transform=ax4.transAxes)
        
        # === FOOTER ===
        fig.text(0.1, 0.01, 'Generated by AI-Powered Fund Analysis Dashboard', 
                fontsize=8, style='italic', alpha=0.7)
        fig.text(0.9, 0.01, 'Page 1 of 1', fontsize=8, alpha=0.7, ha='right')
        
        # Save the figure
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
