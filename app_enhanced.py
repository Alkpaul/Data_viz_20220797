# =============================================================================
# FRENCH STATE BUDGET ANALYSIS WEB APPLICATION - ENHANCED VERSION
# =============================================================================
# This comprehensive application provides advanced analysis and visualization
# of French state budget data from 2012-2022. Features include:
# - Advanced analytics and predictive modeling
# - Interactive dashboards with real-time filtering
# - Comparative analysis and benchmarking
# - Automated insights and recommendations
# - Export capabilities and data management
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import json
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="📊 French State Budget Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main header with gradient and animation */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
        border-right: 3px solid #dee2e6;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #2e86ab);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2e86ab, #1f77b4);
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, #ffffff, #f8f9fa);
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        transform: translateY(-1px);
    }
    
    /* Success/Error styling */
    .stSuccess {
        background: linear-gradient(90deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(90deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #2196f3;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left: 5px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def load_and_preprocess_data():
    """
    Load and preprocess the budget data with comprehensive cleaning and transformation.
    This function handles data loading, cleaning, and preparation for analysis.
    """
    try:
        # Load the cleaned data
        df = pd.read_csv('balances_clean.csv', encoding='utf-8')
        
        # Clean balance values function
        def clean_balance_value(value):
            """Convert balance values to float, handling French number formats"""
            if pd.isna(value) or value == '-' or str(value).strip() == '':
                return np.nan
            cleaned = str(value).strip().replace(' ', '').replace(',', '.')
            try:
                return float(cleaned)
            except ValueError:
                return np.nan
        
        # Get balance columns and clean them
        balance_columns = [col for col in df.columns if 'Balance Sortie' in col]
        for col in balance_columns:
            df[col] = df[col].apply(clean_balance_value)
        
        # Create long format for time series analysis
        df_melted = pd.melt(
            df, 
            id_vars=['Postes', 'Sous-postes', 'Indicateurs de synthèse', 'Indicateurs de détail', 
                     'Compte', 'Nature Budgétaire', 'Programme', 'Libellé Ministère'],
            value_vars=balance_columns,
            var_name='Année',
            value_name='Balance'
        )
        
        # Extract year from column name
        df_melted['Année'] = df_melted['Année'].str.extract(r'(\d{4})').astype(int)
        
        # Add additional calculated columns for analysis
        df_melted['Balance_Millions'] = df_melted['Balance'] / 1e6
        df_melted['Is_Revenue'] = df_melted['Balance'] > 0
        df_melted['Is_Expense'] = df_melted['Balance'] < 0
        df_melted['Abs_Balance'] = abs(df_melted['Balance'])
        
        # Create categories for better analysis
        df_melted['Balance_Category'] = pd.cut(
            df_melted['Balance_Millions'], 
            bins=[-np.inf, -100, -10, 0, 10, 100, np.inf],
            labels=['Major Expense', 'Significant Expense', 'Minor Expense', 
                   'Minor Revenue', 'Significant Revenue', 'Major Revenue']
        )
        
        return df, df_melted, balance_columns
        
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure 'balances_clean.csv' is present.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None

# =============================================================================
# ANALYTICAL FUNCTIONS
# =============================================================================

def calculate_advanced_metrics(df_melted):
    """
    Calculate advanced financial metrics and KPIs for comprehensive analysis.
    Returns a dictionary of calculated metrics.
    """
    metrics = {}
    
    # Basic financial metrics
    metrics['total_balance'] = df_melted['Balance'].sum()
    metrics['avg_balance'] = df_melted['Balance'].mean()
    metrics['median_balance'] = df_melted['Balance'].median()
    metrics['std_balance'] = df_melted['Balance'].std()
    
    # Revenue vs Expense analysis
    revenue_total = df_melted[df_melted['Balance'] > 0]['Balance'].sum()
    expense_total = abs(df_melted[df_melted['Balance'] < 0]['Balance'].sum())
    metrics['revenue_total'] = revenue_total
    metrics['expense_total'] = expense_total
    metrics['net_balance'] = revenue_total - expense_total
    metrics['revenue_expense_ratio'] = revenue_total / expense_total if expense_total > 0 else 0
    
    # Volatility metrics
    yearly_totals = df_melted.groupby('Année')['Balance'].sum()
    metrics['volatility'] = yearly_totals.std()
    metrics['trend'] = yearly_totals.iloc[-1] - yearly_totals.iloc[0]
    
    # Growth metrics
    if len(yearly_totals) > 1:
        metrics['cagr'] = ((yearly_totals.iloc[-1] / yearly_totals.iloc[0]) ** (1/(len(yearly_totals)-1)) - 1) * 100
    
    return metrics

def perform_predictive_analysis(df_melted, years_ahead=3):
    """
    Perform predictive analysis using machine learning models to forecast
    future budget trends based on historical data.
    """
    # Prepare data for prediction
    yearly_data = df_melted.groupby('Année')['Balance'].sum().reset_index()
    yearly_data['Year_Normalized'] = (yearly_data['Année'] - yearly_data['Année'].min()) / (yearly_data['Année'].max() - yearly_data['Année'].min())
    
    X = yearly_data[['Year_Normalized']].values
    y = yearly_data['Balance'].values
    
    # Multiple models for comparison
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    predictions = {}
    model_scores = {}
    
    for name, model in models.items():
        # Fit model
        model.fit(X, y)
        
        # Predict future years
        future_years = np.arange(yearly_data['Année'].max() + 1, yearly_data['Année'].max() + 1 + years_ahead)
        future_years_norm = (future_years - yearly_data['Année'].min()) / (yearly_data['Année'].max() - yearly_data['Année'].min())
        future_predictions = model.predict(future_years_norm.reshape(-1, 1))
        
        predictions[name] = {
            'years': future_years,
            'predictions': future_predictions,
            'r2_score': r2_score(y, model.predict(X)),
            'mae': mean_absolute_error(y, model.predict(X))
        }
    
    return predictions, yearly_data

def generate_insights(df_melted, metrics):
    """
    Generate automated insights and recommendations based on the data analysis.
    Returns a list of insights with their importance levels.
    """
    insights = []
    
    # Financial health insights
    if metrics['net_balance'] > 0:
        insights.append({
            'type': 'positive',
            'title': 'Positive Net Balance',
            'description': f'The overall budget shows a positive net balance of {metrics["net_balance"]/1e6:.1f}M€',
            'importance': 'high'
        })
    else:
        insights.append({
            'type': 'warning',
            'title': 'Negative Net Balance',
            'description': f'The budget shows a deficit of {abs(metrics["net_balance"])/1e6:.1f}M€',
            'importance': 'high'
        })
    
    # Volatility insights
    if metrics['volatility'] > metrics['avg_balance'] * 0.5:
        insights.append({
            'type': 'warning',
            'title': 'High Budget Volatility',
            'description': f'Budget shows high volatility (σ={metrics["volatility"]/1e6:.1f}M€), indicating unstable financial planning',
            'importance': 'medium'
        })
    
    # Trend insights
    if metrics['trend'] > 0:
        insights.append({
            'type': 'positive',
            'title': 'Improving Trend',
            'description': f'Budget shows improving trend with {metrics["trend"]/1e6:.1f}M€ increase over the period',
            'importance': 'medium'
        })
    else:
        insights.append({
            'type': 'warning',
            'title': 'Declining Trend',
            'description': f'Budget shows declining trend with {abs(metrics["trend"])/1e6:.1f}M€ decrease over the period',
            'importance': 'medium'
        })
    
    # Revenue-Expense ratio insights
    if metrics['revenue_expense_ratio'] > 1.1:
        insights.append({
            'type': 'positive',
            'title': 'Healthy Revenue-Expense Ratio',
            'description': f'Revenue exceeds expenses by {(metrics["revenue_expense_ratio"]-1)*100:.1f}%',
            'importance': 'medium'
        })
    elif metrics['revenue_expense_ratio'] < 0.9:
        insights.append({
            'type': 'warning',
            'title': 'Concerning Revenue-Expense Ratio',
            'description': f'Expenses exceed revenue by {(1-metrics["revenue_expense_ratio"])*100:.1f}%',
            'importance': 'high'
        })
    
    return insights

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_advanced_time_series(df_melted, selected_years):
    """
    Create advanced time series visualization with multiple components
    including trend analysis, seasonality, and forecasting.
    """
    # Filter data
    df_filtered = df_melted[df_melted['Année'].isin(selected_years)]
    yearly_data = df_filtered.groupby('Année')['Balance'].sum().reset_index()
    
    # Create subplot with multiple traces
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Budget Evolution', 'Revenue vs Expenses', 'Budget Distribution', 'Trend Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Main budget evolution
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Année'],
            y=yearly_data['Balance'] / 1e6,
            mode='lines+markers',
            name='Total Budget',
            line=dict(width=4, color='#1f77b4'),
            marker=dict(size=10)
        ),
        row=1, col=1
    )
    
    # Revenue vs Expenses
    revenue_data = df_filtered[df_filtered['Balance'] > 0].groupby('Année')['Balance'].sum() / 1e6
    expense_data = abs(df_filtered[df_filtered['Balance'] < 0].groupby('Année')['Balance'].sum()) / 1e6
    
    fig.add_trace(
        go.Bar(x=revenue_data.index, y=revenue_data.values, name='Revenue', marker_color='#2ca02c'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=expense_data.index, y=expense_data.values, name='Expenses', marker_color='#d62728'),
        row=1, col=2
    )
    
    # Budget distribution
    fig.add_trace(
        go.Histogram(
            x=df_filtered['Balance'] / 1e6,
            nbinsx=50,
            name='Distribution',
            marker_color='#ff7f0e',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Trend analysis with moving average
    yearly_data['MA_3'] = yearly_data['Balance'].rolling(window=3, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Année'],
            y=yearly_data['MA_3'] / 1e6,
            mode='lines',
            name='3-Year Moving Average',
            line=dict(width=3, color='#9467bd', dash='dash')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Advanced Budget Analysis Dashboard",
        showlegend=True
    )
    
    return fig

def create_correlation_heatmap(df_melted, selected_years):
    """
    Create correlation heatmap for different budget categories and time periods.
    """
    df_filtered = df_melted[df_melted['Année'].isin(selected_years)]
    
    # Create pivot table for correlation analysis
    pivot_data = df_filtered.pivot_table(
        values='Balance',
        index='Année',
        columns='Postes',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate correlation matrix
    correlation_matrix = pivot_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=[col[:30] + '...' if len(col) > 30 else col for col in correlation_matrix.columns],
        y=[col[:30] + '...' if len(col) > 30 else col for col in correlation_matrix.columns],
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Budget Categories Correlation Matrix",
        xaxis_title="Budget Categories",
        yaxis_title="Budget Categories",
        height=600,
        width=800
    )
    
    return fig

def create_ministry_performance_dashboard(df_melted, selected_years):
    """
    Create comprehensive ministry performance dashboard with multiple visualizations.
    """
    df_filtered = df_melted[df_melted['Année'].isin(selected_years)]
    
    # Calculate ministry performance metrics
    ministry_metrics = df_filtered.groupby('Libellé Ministère').agg({
        'Balance': ['sum', 'mean', 'std', 'count'],
        'Balance_Millions': 'sum'
    }).round(2)
    
    ministry_metrics.columns = ['Total_Balance', 'Avg_Balance', 'Std_Balance', 'Count', 'Total_Millions']
    ministry_metrics = ministry_metrics.sort_values('Total_Millions', ascending=False).head(15)
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Budget by Ministry', 'Average Budget by Ministry', 
                       'Budget Volatility by Ministry', 'Budget Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Total budget
    fig.add_trace(
        go.Bar(
            x=ministry_metrics['Total_Millions'],
            y=ministry_metrics.index,
            orientation='h',
            name='Total Budget',
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    
    # Average budget
    fig.add_trace(
        go.Bar(
            x=ministry_metrics['Avg_Balance'] / 1e6,
            y=ministry_metrics.index,
            orientation='h',
            name='Average Budget',
            marker_color='#ff7f0e'
        ),
        row=1, col=2
    )
    
    # Volatility
    fig.add_trace(
        go.Bar(
            x=ministry_metrics['Std_Balance'] / 1e6,
            y=ministry_metrics.index,
            orientation='h',
            name='Volatility',
            marker_color='#d62728'
        ),
        row=2, col=1
    )
    
    # Distribution
    fig.add_trace(
        go.Histogram(
            x=ministry_metrics['Total_Millions'],
            nbinsx=20,
            name='Distribution',
            marker_color='#2ca02c',
            opacity=0.7
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Ministry Performance Dashboard",
        showlegend=False
    )
    
    return fig, ministry_metrics

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application function that orchestrates the entire dashboard.
    """
    # Header
    st.markdown('<h1 class="main-header">📊 French State Budget Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Analysis of French State Budget Data (2012-2022)")
    
    # Load data
    with st.spinner("🔄 Loading and preprocessing data..."):
        df, df_melted, balance_columns = load_and_preprocess_data()
    
    if df is None:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Filters & Configuration")
    
    # Year selection
    years = sorted(df_melted['Année'].unique())
    selected_years = st.sidebar.multiselect(
        "📅 Select Years",
        options=years,
        default=years,
        help="Choose the years to analyze"
    )
    
    # Ministry selection
    ministeres = sorted([m for m in df_melted['Libellé Ministère'].unique() if pd.notna(m)])
    selected_ministeres = st.sidebar.multiselect(
        "🏛️ Select Ministries",
        options=ministeres,
        default=ministeres[:8],
        help="Choose ministries to analyze"
    )
    
    # Budget category selection
    postes = sorted(df_melted['Postes'].unique())
    selected_postes = st.sidebar.multiselect(
        "💰 Select Budget Categories",
        options=postes,
        default=postes[:12],
        help="Choose budget categories to analyze"
    )
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "📊 Analysis Type",
        options=["Comprehensive", "Financial Health", "Trend Analysis", "Comparative"],
        help="Choose the type of analysis to perform"
    )
    
    # Apply filters
    df_filtered = df_melted[
        (df_melted['Année'].isin(selected_years)) &
        (df_melted['Libellé Ministère'].isin(selected_ministeres)) &
        (df_melted['Postes'].isin(selected_postes))
    ]
    
    # Calculate metrics
    metrics = calculate_advanced_metrics(df_filtered)
    
    # Display key metrics
    st.header("📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="💰 Total Balance",
            value=f"{metrics['total_balance']/1e6:,.0f} M€",
            delta=f"{metrics['trend']/1e6:+,.0f} M€ trend"
        )
    
    with col2:
        st.metric(
            label="📊 Net Balance",
            value=f"{metrics['net_balance']/1e6:,.0f} M€",
            delta=f"{(metrics['revenue_expense_ratio']-1)*100:+.1f}% ratio"
        )
    
    with col3:
        st.metric(
            label="📈 Volatility",
            value=f"{metrics['volatility']/1e6:,.0f} M€",
            delta="Standard deviation"
        )
    
    with col4:
        st.metric(
            label="🎯 Revenue Total",
            value=f"{metrics['revenue_total']/1e6:,.0f} M€",
            delta="Positive balances"
        )
    
    with col5:
        st.metric(
            label="💸 Expense Total",
            value=f"{metrics['expense_total']/1e6:,.0f} M€",
            delta="Negative balances"
        )
    
    # Generate and display insights
    insights = generate_insights(df_filtered, metrics)
    
    if insights:
        st.header("💡 Automated Insights & Recommendations")
        
        for insight in insights:
            if insight['importance'] == 'high':
                if insight['type'] == 'positive':
                    st.success(f"✅ **{insight['title']}**: {insight['description']}")
                else:
                    st.error(f"⚠️ **{insight['title']}**: {insight['description']}")
            else:
                if insight['type'] == 'positive':
                    st.info(f"ℹ️ **{insight['title']}**: {insight['description']}")
                else:
                    st.warning(f"⚠️ **{insight['title']}**: {insight['description']}")
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview Dashboard", 
        "🏛️ Ministry Analysis", 
        "💰 Budget Categories", 
        "📈 Time Series Analysis",
        "🔮 Predictive Analytics",
        "🔍 Advanced Analytics"
    ])
    
    with tab1:
        st.header("📊 Comprehensive Overview Dashboard")
        
        # Advanced time series
        fig_overview = create_advanced_time_series(df_filtered, selected_years)
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Summary Statistics")
            summary_stats = df_filtered['Balance'].describe()
            st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            st.subheader("📊 Budget Categories Distribution")
            category_counts = df_filtered['Balance_Category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Budget Categories Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.header("🏛️ Ministry Performance Analysis")
        
        # Ministry performance dashboard
        fig_ministry, ministry_metrics = create_ministry_performance_dashboard(df_filtered, selected_years)
        st.plotly_chart(fig_ministry, use_container_width=True)
        
        # Ministry comparison table
        st.subheader("📋 Ministry Performance Metrics")
        st.dataframe(ministry_metrics, use_container_width=True)
        
        # Ministry evolution over time
        if len(selected_ministeres) <= 5:
            st.subheader("📈 Ministry Evolution Over Time")
            ministry_evolution = df_filtered.groupby(['Année', 'Libellé Ministère'])['Balance'].sum().unstack(fill_value=0)
            
            fig_evolution = go.Figure()
            for ministry in ministry_evolution.columns:
                fig_evolution.add_trace(go.Scatter(
                    x=ministry_evolution.index,
                    y=ministry_evolution[ministry] / 1e6,
                    mode='lines+markers',
                    name=ministry,
                    line=dict(width=3)
                ))
            
            fig_evolution.update_layout(
                title="Ministry Budget Evolution",
                xaxis_title="Year",
                yaxis_title="Budget (Millions €)",
                height=500
            )
            st.plotly_chart(fig_evolution, use_container_width=True)
    
    with tab3:
        st.header("💰 Budget Categories Analysis")
        
        # Budget categories heatmap
        if len(selected_postes) <= 20:
            fig_heatmap = create_correlation_heatmap(df_filtered, selected_years)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Revenue vs Expense analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💚 Top Revenue Categories")
            revenue_categories = df_filtered[df_filtered['Balance'] > 0].groupby('Postes')['Balance'].sum().sort_values(ascending=False).head(10)
            
            fig_revenue = px.bar(
                x=revenue_categories.values / 1e6,
                y=revenue_categories.index,
                orientation='h',
                title="Top 10 Revenue Categories",
                labels={'x': 'Revenue (Millions €)', 'y': 'Budget Categories'},
                color=revenue_categories.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            st.subheader("❤️ Top Expense Categories")
            expense_categories = df_filtered[df_filtered['Balance'] < 0].groupby('Postes')['Balance'].sum().sort_values(ascending=True).head(10)
            
            fig_expense = px.bar(
                x=abs(expense_categories.values) / 1e6,
                y=expense_categories.index,
                orientation='h',
                title="Top 10 Expense Categories",
                labels={'x': 'Expenses (Millions €)', 'y': 'Budget Categories'},
                color=abs(expense_categories.values),
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_expense, use_container_width=True)
    
    with tab4:
        st.header("📈 Advanced Time Series Analysis")
        
        # Time series decomposition
        yearly_data = df_filtered.groupby('Année')['Balance'].sum().reset_index()
        
        # Trend analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Trend Analysis")
            
            # Linear trend
            x = np.arange(len(yearly_data))
            y = yearly_data['Balance'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=yearly_data['Année'],
                y=yearly_data['Balance'] / 1e6,
                mode='lines+markers',
                name='Actual Data',
                line=dict(width=3, color='#1f77b4')
            ))
            
            # Add trend line
            trend_line = slope * x + intercept
            fig_trend.add_trace(go.Scatter(
                x=yearly_data['Année'],
                y=trend_line / 1e6,
                mode='lines',
                name=f'Trend (R²={r_value**2:.3f})',
                line=dict(width=2, color='red', dash='dash')
            ))
            
            fig_trend.update_layout(
                title="Budget Trend Analysis",
                xaxis_title="Year",
                yaxis_title="Budget (Millions €)"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            st.subheader("📈 Growth Analysis")
            
            # Calculate year-over-year growth
            yearly_data['YoY_Growth'] = yearly_data['Balance'].pct_change() * 100
            yearly_data['Cumulative_Growth'] = ((yearly_data['Balance'] / yearly_data['Balance'].iloc[0]) - 1) * 100
            
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Bar(
                x=yearly_data['Année'],
                y=yearly_data['YoY_Growth'],
                name='Year-over-Year Growth (%)',
                marker_color='#ff7f0e'
            ))
            fig_growth.add_trace(go.Scatter(
                x=yearly_data['Année'],
                y=yearly_data['Cumulative_Growth'],
                mode='lines+markers',
                name='Cumulative Growth (%)',
                line=dict(width=3, color='#2ca02c'),
                yaxis='y2'
            ))
            
            fig_growth.update_layout(
                title="Budget Growth Analysis",
                xaxis_title="Year",
                yaxis_title="YoY Growth (%)",
                yaxis2=dict(
                    title="Cumulative Growth (%)",
                    overlaying='y',
                    side='right'
                )
            )
            st.plotly_chart(fig_growth, use_container_width=True)
    
    with tab5:
        st.header("🔮 Predictive Analytics & Forecasting")
        
        # Perform predictive analysis
        with st.spinner("🤖 Training machine learning models..."):
            predictions, historical_data = perform_predictive_analysis(df_filtered, years_ahead=5)
        
        # Display model comparison
        st.subheader("🤖 Model Performance Comparison")
        
        model_comparison = pd.DataFrame({
            'Model': list(predictions.keys()),
            'R² Score': [pred['r2_score'] for pred in predictions.values()],
            'Mean Absolute Error': [pred['mae']/1e6 for pred in predictions.values()]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(model_comparison, use_container_width=True)
        
        with col2:
            fig_models = px.bar(
                x=model_comparison['Model'],
                y=model_comparison['R² Score'],
                title="Model R² Score Comparison",
                labels={'x': 'Model', 'y': 'R² Score'}
            )
            st.plotly_chart(fig_models, use_container_width=True)
        
        # Display predictions
        st.subheader("🔮 Budget Forecast (Next 5 Years)")
        
        # Create forecast visualization
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=historical_data['Année'],
            y=historical_data['Balance'] / 1e6,
            mode='lines+markers',
            name='Historical Data',
            line=dict(width=3, color='#1f77b4')
        ))
        
        # Predictions for each model
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        for i, (model_name, pred_data) in enumerate(predictions.items()):
            fig_forecast.add_trace(go.Scatter(
                x=pred_data['years'],
                y=pred_data['predictions'] / 1e6,
                mode='lines+markers',
                name=f'{model_name} Forecast',
                line=dict(width=2, color=colors[i], dash='dash')
            ))
        
        fig_forecast.update_layout(
            title="Budget Forecast Comparison",
            xaxis_title="Year",
            yaxis_title="Budget (Millions €)",
            height=500
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast details
        st.subheader("📋 Detailed Forecasts")
        for model_name, pred_data in predictions.items():
            with st.expander(f"🔮 {model_name} Forecast Details"):
                forecast_df = pd.DataFrame({
                    'Year': pred_data['years'],
                    'Predicted Budget (M€)': pred_data['predictions'] / 1e6,
                    'Model R²': pred_data['r2_score'],
                    'MAE (M€)': pred_data['mae'] / 1e6
                })
                st.dataframe(forecast_df, use_container_width=True)
    
    with tab6:
        st.header("🔍 Advanced Analytics & Deep Dive")
        
        # Statistical analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistical Analysis")
            
            # Normality test
            from scipy.stats import shapiro
            sample_data = df_filtered['Balance'].dropna().sample(min(5000, len(df_filtered)))
            stat, p_value = shapiro(sample_data)
            
            st.write(f"**Shapiro-Wilk Normality Test:**")
            st.write(f"- Statistic: {stat:.4f}")
            st.write(f"- p-value: {p_value:.4f}")
            st.write(f"- Normal distribution: {'Yes' if p_value > 0.05 else 'No'}")
            
            # Outlier detection
            Q1 = df_filtered['Balance'].quantile(0.25)
            Q3 = df_filtered['Balance'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df_filtered[(df_filtered['Balance'] < Q1 - 1.5*IQR) | (df_filtered['Balance'] > Q3 + 1.5*IQR)]
            
            st.write(f"**Outlier Analysis:**")
            st.write(f"- Number of outliers: {len(outliers)}")
            st.write(f"- Percentage: {len(outliers)/len(df_filtered)*100:.2f}%")
        
        with col2:
            st.subheader("📈 Advanced Visualizations")
            
            # Box plot by year
            fig_box = px.box(
                df_filtered,
                x='Année',
                y='Balance_Millions',
                title="Budget Distribution by Year",
                labels={'Balance_Millions': 'Budget (Millions €)', 'Année': 'Year'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        # Create correlation matrix for numerical variables
        numerical_cols = ['Balance', 'Balance_Millions', 'Abs_Balance']
        correlation_data = df_filtered[numerical_cols + ['Année']].corr()
        
        fig_corr = px.imshow(
            correlation_data,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data export section
        st.subheader("💾 Data Export & Download")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export filtered data
            csv = df_filtered.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"budget_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export metrics
            metrics_json = json.dumps(metrics, indent=2, default=str)
            st.download_button(
                label="📊 Download Metrics (JSON)",
                data=metrics_json,
                file_name=f"budget_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Export insights
            insights_json = json.dumps(insights, indent=2, default=str)
            st.download_button(
                label="💡 Download Insights (JSON)",
                data=insights_json,
                file_name=f"budget_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>📊 French State Budget Analysis Dashboard</h3>
        <p>Comprehensive analysis of French state budget data (2012-2022)</p>
        <p>Built with Streamlit, Plotly, and advanced machine learning</p>
        <p><strong>Features:</strong> Predictive Analytics • Interactive Visualizations • Automated Insights • Export Capabilities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
