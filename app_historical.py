# =============================================================================
# FRENCH STATE BUDGET ANALYSIS - HISTORICAL STORYTELLING DASHBOARD
# =============================================================================
# This application tells the story of France from 2012-2022 through its budget data,
# connecting financial decisions to historical events, elections, and national crises.
# Features include historical analysis, election correlations, and future predictions.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import json
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="📊 France Through Its Budget - Historical Analysis",
    page_icon="🇫🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main header with French flag colors */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #002395, #ffffff, #ed2939);
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
    
    /* Historical event cards */
    .event-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #002395;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .event-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Crisis indicators */
    .crisis-indicator {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Growth indicators */
    .growth-indicator {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Election indicators */
    .election-indicator {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left: 5px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HISTORICAL EVENTS DATA
# =============================================================================

FRENCH_HISTORICAL_EVENTS = {
    2012: {
        'elections': 'Presidential Election - François Hollande elected',
        'key_events': ['End of Sarkozy presidency', 'Socialist Party victory'],
        'budget_impact': 'Shift towards social spending, tax increases'
    },
    2013: {
        'key_events': ['Same-sex marriage legalized', 'Tax increases implemented'],
        'budget_impact': 'Social reforms, increased taxation'
    },
    2014: {
        'key_events': ['Municipal elections', 'Economic recovery begins'],
        'budget_impact': 'Local government funding changes'
    },
    2015: {
        'crisis': 'Terrorist attacks (Charlie Hebdo, Bataclan)',
        'key_events': ['State of emergency declared', 'Security measures increased'],
        'budget_impact': 'Massive increase in defense and security spending'
    },
    2016: {
        'key_events': ['State of emergency extended', 'Labor law reforms'],
        'budget_impact': 'Continued security spending, labor reforms'
    },
    2017: {
        'elections': 'Presidential Election - Emmanuel Macron elected',
        'key_events': ['En Marche! victory', 'Political renewal'],
        'budget_impact': 'New economic policies, digital transformation'
    },
    2018: {
        'key_events': ['Yellow Vests movement begins', 'Tax reforms'],
        'budget_impact': 'Social unrest, tax policy adjustments'
    },
    2019: {
        'key_events': ['Yellow Vests continue', 'Climate protests'],
        'budget_impact': 'Social spending increases, climate initiatives'
    },
    2020: {
        'crisis': 'COVID-19 Pandemic',
        'key_events': ['Lockdown measures', 'Economic stimulus'],
        'budget_impact': 'Massive health spending, economic support'
    },
    2021: {
        'key_events': ['Vaccination campaign', 'Economic recovery'],
        'budget_impact': 'Health infrastructure, economic stimulus'
    },
    2022: {
        'elections': 'Presidential Election - Emmanuel Macron re-elected',
        'key_events': ['War in Ukraine', 'Energy crisis'],
        'budget_impact': 'Defense spending, energy support'
    }
}

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def load_and_preprocess_data():
    """
    Load and preprocess the budget data with historical context.
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
        
        # Add historical context
        df_melted['Historical_Event'] = df_melted['Année'].map(lambda x: FRENCH_HISTORICAL_EVENTS.get(x, {}).get('crisis', ''))
        df_melted['Election_Year'] = df_melted['Année'].map(lambda x: 'Election' if 'elections' in FRENCH_HISTORICAL_EVENTS.get(x, {}) else '')
        
        # Add calculated columns
        df_melted['Balance_Millions'] = df_melted['Balance'] / 1e6
        df_melted['Is_Revenue'] = df_melted['Balance'] > 0
        df_melted['Is_Expense'] = df_melted['Balance'] < 0
        df_melted['Abs_Balance'] = abs(df_melted['Balance'])
        
        return df, df_melted, balance_columns
        
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure 'balances_clean.csv' is present.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None

# =============================================================================
# HISTORICAL ANALYSIS FUNCTIONS
# =============================================================================

def analyze_crisis_impact(df_melted, crisis_year, ministry_focus=None):
    """
    Analyze the impact of a specific crisis on budget allocations.
    """
    # Get data for crisis year and previous year
    crisis_data = df_melted[df_melted['Année'] == crisis_year]
    prev_year_data = df_melted[df_melted['Année'] == crisis_year - 1]
    
    if ministry_focus:
        crisis_data = crisis_data[crisis_data['Libellé Ministère'] == ministry_focus]
        prev_year_data = prev_year_data[prev_year_data['Libellé Ministère'] == ministry_focus]
    
    # Calculate changes
    crisis_total = crisis_data['Balance'].sum()
    prev_total = prev_year_data['Balance'].sum()
    change = crisis_total - prev_total
    change_pct = (change / abs(prev_total)) * 100 if prev_total != 0 else 0
    
    return {
        'crisis_year': crisis_year,
        'crisis_total': crisis_total,
        'prev_total': prev_total,
        'change': change,
        'change_pct': change_pct,
        'crisis_event': FRENCH_HISTORICAL_EVENTS.get(crisis_year, {}).get('crisis', '')
    }

def analyze_election_impact(df_melted, election_years=[2012, 2017, 2022]):
    """
    Analyze budget changes around election years.
    """
    election_analysis = {}
    
    for year in election_years:
        if year in election_years:
            # Get data for election year and previous year
            election_data = df_melted[df_melted['Année'] == year]
            prev_data = df_melted[df_melted['Année'] == year - 1]
            
            # Calculate changes by ministry
            election_by_ministry = election_data.groupby('Libellé Ministère')['Balance'].sum()
            prev_by_ministry = prev_data.groupby('Libellé Ministère')['Balance'].sum()
            
            changes = {}
            for ministry in election_by_ministry.index:
                if ministry in prev_by_ministry.index:
                    change = election_by_ministry[ministry] - prev_by_ministry[ministry]
                    changes[ministry] = change
            
            election_analysis[year] = {
                'total_change': election_data['Balance'].sum() - prev_data['Balance'].sum(),
                'ministry_changes': changes,
                'election_info': FRENCH_HISTORICAL_EVENTS.get(year, {}).get('elections', '')
            }
    
    return election_analysis

def create_historical_timeline(df_melted):
    """
    Create a comprehensive historical timeline with budget impacts.
    """
    timeline_data = []
    
    for year in sorted(df_melted['Année'].unique()):
        year_data = df_melted[df_melted['Année'] == year]
        
        # Calculate key metrics
        total_balance = year_data['Balance'].sum()
        defense_spending = year_data[
            (year_data['Libellé Ministère'].str.contains('Défense', na=False)) |
            (year_data['Postes'].str.contains('défense|sécurité', case=False, na=False))
        ]['Balance'].sum()
        
        health_spending = year_data[
            (year_data['Libellé Ministère'].str.contains('Santé', na=False)) |
            (year_data['Postes'].str.contains('santé|médical', case=False, na=False))
        ]['Balance'].sum()
        
        social_spending = year_data[
            (year_data['Libellé Ministère'].str.contains('Social|Travail', na=False)) |
            (year_data['Postes'].str.contains('social|travail', case=False, na=False))
        ]['Balance'].sum()
        
        timeline_data.append({
            'year': year,
            'total_balance': total_balance,
            'defense_spending': defense_spending,
            'health_spending': health_spending,
            'social_spending': social_spending,
            'events': FRENCH_HISTORICAL_EVENTS.get(year, {}),
            'is_election': 'elections' in FRENCH_HISTORICAL_EVENTS.get(year, {}),
            'is_crisis': 'crisis' in FRENCH_HISTORICAL_EVENTS.get(year, {})
        })
    
    return timeline_data

# =============================================================================
# PREDICTIVE ANALYSIS FUNCTIONS
# =============================================================================

def predict_future_budget(df_melted, years_ahead=5):
    """
    Predict future budget trends based on historical patterns and events.
    """
    # Prepare data
    yearly_data = df_melted.groupby('Année')['Balance'].sum().reset_index()
    yearly_data['Year_Normalized'] = (yearly_data['Année'] - yearly_data['Année'].min()) / (yearly_data['Année'].max() - yearly_data['Année'].min())
    
    X = yearly_data[['Year_Normalized']].values
    y = yearly_data['Balance'].values
    
    # Train multiple models
    models = {
        'Linear Trend': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    predictions = {}
    
    for name, model in models.items():
        model.fit(X, y)
        
        # Predict future years
        future_years = np.arange(yearly_data['Année'].max() + 1, yearly_data['Année'].max() + 1 + years_ahead)
        future_years_norm = (future_years - yearly_data['Année'].min()) / (yearly_data['Année'].max() - yearly_data['Année'].min())
        future_predictions = model.predict(future_years_norm.reshape(-1, 1))
        
        predictions[name] = {
            'years': future_years,
            'predictions': future_predictions,
            'r2_score': r2_score(y, model.predict(X))
        }
    
    return predictions, yearly_data

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_historical_story_chart(df_melted, selected_years):
    """
    Create a comprehensive chart that tells France's story through budget data.
    """
    timeline_data = create_historical_timeline(df_melted)
    timeline_df = pd.DataFrame(timeline_data)
    
    # Filter by selected years
    timeline_df = timeline_df[timeline_df['year'].isin(selected_years)]
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('France Budget Evolution (2012-2022)', 'Defense Spending Timeline', 
                       'Health & Social Spending', 'Election Impact Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Main budget evolution with events
    fig.add_trace(
        go.Scatter(
            x=timeline_df['year'],
            y=timeline_df['total_balance'] / 1e6,
            mode='lines+markers',
            name='Total Budget',
            line=dict(width=4, color='#002395'),
            marker=dict(size=10)
        ),
        row=1, col=1
    )
    
    # Add crisis indicators
    crisis_years = timeline_df[timeline_df['is_crisis']]['year']
    crisis_values = timeline_df[timeline_df['is_crisis']]['total_balance'] / 1e6
    
    fig.add_trace(
        go.Scatter(
            x=crisis_years,
            y=crisis_values,
            mode='markers',
            name='Crisis Years',
            marker=dict(size=15, color='red', symbol='x'),
            text=['2015: Terrorist Attacks', '2020: COVID-19 Pandemic', '2022: Ukraine War'],
            textposition='top center'
        ),
        row=1, col=1
    )
    
    # Defense spending timeline
    fig.add_trace(
        go.Scatter(
            x=timeline_df['year'],
            y=timeline_df['defense_spending'] / 1e6,
            mode='lines+markers',
            name='Defense Spending',
            line=dict(width=3, color='#dc3545'),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Health and Social spending
    fig.add_trace(
        go.Scatter(
            x=timeline_df['year'],
            y=timeline_df['health_spending'] / 1e6,
            mode='lines+markers',
            name='Health Spending',
            line=dict(width=3, color='#28a745'),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timeline_df['year'],
            y=timeline_df['social_spending'] / 1e6,
            mode='lines+markers',
            name='Social Spending',
            line=dict(width=3, color='#ffc107'),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Election impact
    election_years = timeline_df[timeline_df['is_election']]['year']
    election_changes = []
    
    for year in election_years:
        if year > timeline_df['year'].min():
            prev_year_data = timeline_df[timeline_df['year'] == year - 1]
            current_year_data = timeline_df[timeline_df['year'] == year]
            if not prev_year_data.empty and not current_year_data.empty:
                change = (current_year_data['total_balance'].iloc[0] - prev_year_data['total_balance'].iloc[0]) / 1e6
                election_changes.append(change)
            else:
                election_changes.append(0)
        else:
            election_changes.append(0)
    
    fig.add_trace(
        go.Bar(
            x=election_years,
            y=election_changes,
            name='Election Year Changes',
            marker_color='#6f42c1'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="🇫🇷 France's Story Through Budget Data (2012-2022)",
        showlegend=True
    )
    
    return fig

def create_crisis_analysis_chart(df_melted, crisis_year=2015):
    """
    Create detailed analysis of crisis impact on budget.
    """
    crisis_analysis = analyze_crisis_impact(df_melted, crisis_year)
    
    # Get ministry-level changes
    crisis_data = df_melted[df_melted['Année'] == crisis_year]
    prev_data = df_melted[df_melted['Année'] == crisis_year - 1]
    
    ministry_changes = {}
    for ministry in crisis_data['Libellé Ministère'].unique():
        if pd.notna(ministry):
            crisis_ministry = crisis_data[crisis_data['Libellé Ministère'] == ministry]['Balance'].sum()
            prev_ministry = prev_data[prev_data['Libellé Ministère'] == ministry]['Balance'].sum()
            change = crisis_ministry - prev_ministry
            ministry_changes[ministry] = change
    
    # Sort by change magnitude
    sorted_changes = sorted(ministry_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    ministries, changes = zip(*sorted_changes)
    
    fig = go.Figure()
    
    colors = ['red' if change < 0 else 'green' for change in changes]
    
    fig.add_trace(go.Bar(
        x=list(ministries),
        y=[change/1e6 for change in changes],
        marker_color=colors,
        text=[f'{change/1e6:+.1f}M€' for change in changes],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"🏛️ Ministry Budget Changes During {crisis_year} Crisis",
        xaxis_title="Ministries",
        yaxis_title="Budget Change (Millions €)",
        height=500
    )
    
    return fig, crisis_analysis

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application function focused on historical storytelling.
    """
    # Header
    st.markdown('<h1 class="main-header">🇫🇷 France Through Its Budget</h1>', unsafe_allow_html=True)
    st.markdown("### Telling France's Story Through Budget Data (2012-2022)")
    
    # Load data
    with st.spinner("🔄 Loading historical budget data..."):
        df, df_melted, balance_columns = load_and_preprocess_data()
    
    if df is None:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Analysis Configuration")
    
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
    
    # Crisis analysis selection
    crisis_years = [2015, 2020, 2022]  # Major crisis years
    selected_crisis = st.sidebar.selectbox(
        "🚨 Crisis Analysis",
        options=crisis_years,
        help="Select crisis year for detailed analysis"
    )
    
    # Apply filters
    df_filtered = df_melted[
        (df_melted['Année'].isin(selected_years)) &
        (df_melted['Libellé Ministère'].isin(selected_ministeres))
    ]
    
    # Calculate key metrics
    total_balance = df_filtered['Balance'].sum()
    avg_balance = df_filtered['Balance'].mean()
    
    # Display key metrics
    st.header("📈 France's Financial Journey")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Budget",
            value=f"{total_balance/1e6:,.0f} M€",
            delta=f"Over {len(selected_years)} years"
        )
    
    with col2:
        st.metric(
            label="📊 Average Balance",
            value=f"{avg_balance/1e6:,.1f} M€",
            delta="Per entry"
        )
    
    with col3:
        crisis_count = len([y for y in selected_years if 'crisis' in FRENCH_HISTORICAL_EVENTS.get(y, {})])
        st.metric(
            label="🚨 Crisis Years",
            value=f"{crisis_count}",
            delta="Major events"
        )
    
    with col4:
        election_count = len([y for y in selected_years if 'elections' in FRENCH_HISTORICAL_EVENTS.get(y, {})])
        st.metric(
            label="🗳️ Election Years",
            value=f"{election_count}",
            delta="Presidential elections"
        )
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Historical Story", 
        "🏛️ Ministry Analysis", 
        "🚨 Crisis Impact", 
        "🗳️ Election Analysis",
        "🔮 Future Predictions"
    ])
    
    with tab1:
        st.header("📖 France's Story Through Budget Data")
        
        # Historical timeline chart
        fig_story = create_historical_story_chart(df_filtered, selected_years)
        st.plotly_chart(fig_story, use_container_width=True)
        
        # Historical events summary
        st.subheader("📅 Key Historical Events & Budget Impact")
        
        for year in selected_years:
            events = FRENCH_HISTORICAL_EVENTS.get(year, {})
            if events:
                with st.expander(f"📅 {year} - {events.get('crisis', events.get('elections', 'Key Events'))}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Events:**")
                        if 'crisis' in events:
                            st.markdown(f"🚨 **Crisis:** {events['crisis']}")
                        if 'elections' in events:
                            st.markdown(f"🗳️ **Elections:** {events['elections']}")
                        if 'key_events' in events:
                            for event in events['key_events']:
                                st.write(f"• {event}")
                    
                    with col2:
                        st.write("**Budget Impact:**")
                        if 'budget_impact' in events:
                            st.write(events['budget_impact'])
                        
                        # Show actual budget data for this year
                        year_data = df_filtered[df_filtered['Année'] == year]
                        if not year_data.empty:
                            total_year = year_data['Balance'].sum()
                            st.write(f"**Total Budget:** {total_year/1e6:,.0f} M€")
    
    with tab2:
        st.header("🏛️ Ministry Performance Analysis")
        
        # Ministry performance over time
        ministry_evolution = df_filtered.groupby(['Année', 'Libellé Ministère'])['Balance'].sum().unstack(fill_value=0)
        
        # Top 10 ministries by total budget
        ministry_totals = df_filtered.groupby('Libellé Ministère')['Balance'].sum().sort_values(ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Top 10 Ministries by Total Budget")
            fig_ministries = px.bar(
                x=ministry_totals.values / 1e6,
                y=ministry_totals.index,
                orientation='h',
                title="Ministry Budget Totals (2012-2022)",
                labels={'x': 'Total Budget (Millions €)', 'y': 'Ministries'},
                color=ministry_totals.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_ministries, use_container_width=True)
        
        with col2:
            st.subheader("📈 Ministry Evolution Over Time")
            if len(selected_ministeres) <= 5:
                fig_evolution = go.Figure()
                for ministry in selected_ministeres[:5]:
                    if ministry in ministry_evolution.columns:
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
                    height=400
                )
                st.plotly_chart(fig_evolution, use_container_width=True)
            else:
                st.info("Select 5 or fewer ministries to see evolution chart")
        
        # Ministry performance metrics
        st.subheader("📋 Ministry Performance Metrics")
        ministry_metrics = df_filtered.groupby('Libellé Ministère').agg({
            'Balance': ['sum', 'mean', 'std', 'count']
        }).round(2)
        ministry_metrics.columns = ['Total_Budget', 'Average_Budget', 'Volatility', 'Entries']
        ministry_metrics = ministry_metrics.sort_values('Total_Budget', ascending=False)
        st.dataframe(ministry_metrics, use_container_width=True)
    
    with tab3:
        st.header("🚨 Crisis Impact Analysis")
        
        # Crisis analysis for selected year
        fig_crisis, crisis_analysis = create_crisis_analysis_chart(df_filtered, selected_crisis)
        
        st.plotly_chart(fig_crisis, use_container_width=True)
        
        # Crisis details
        st.subheader(f"📊 {selected_crisis} Crisis Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="crisis-indicator">
                <h4>🚨 Crisis: {crisis_analysis['crisis_event']}</h4>
                <p><strong>Budget Change:</strong> {crisis_analysis['change']/1e6:+,.1f} M€</p>
                <p><strong>Percentage Change:</strong> {crisis_analysis['change_pct']:+.1f}%</p>
                <p><strong>Crisis Year Total:</strong> {crisis_analysis['crisis_total']/1e6:,.1f} M€</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Show specific ministry impacts
            crisis_data = df_filtered[df_filtered['Année'] == selected_crisis]
            prev_data = df_filtered[df_filtered['Année'] == selected_crisis - 1]
            
            ministry_impacts = {}
            for ministry in crisis_data['Libellé Ministère'].unique():
                if pd.notna(ministry):
                    crisis_ministry = crisis_data[crisis_data['Libellé Ministère'] == ministry]['Balance'].sum()
                    prev_ministry = prev_data[prev_data['Libellé Ministère'] == ministry]['Balance'].sum()
                    change = crisis_ministry - prev_ministry
                    ministry_impacts[ministry] = change
            
            # Top 5 impacts
            top_impacts = sorted(ministry_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            st.write("**Top 5 Ministry Impacts:**")
            for ministry, change in top_impacts:
                color = "🔴" if change < 0 else "🟢"
                st.write(f"{color} {ministry}: {change/1e6:+.1f} M€")
    
    with tab4:
        st.header("🗳️ Election Impact Analysis")
        
        # Analyze election impacts
        election_analysis = analyze_election_impact(df_filtered)
        
        st.subheader("📊 Budget Changes Around Elections")
        
        election_data = []
        for year, data in election_analysis.items():
            election_data.append({
                'Year': year,
                'Total_Change': data['total_change'] / 1e6,
                'Election': data['election_info']
            })
        
        election_df = pd.DataFrame(election_data)
        
        fig_elections = px.bar(
            election_df,
            x='Year',
            y='Total_Change',
            title="Budget Changes During Election Years",
            labels={'Total_Change': 'Budget Change (Millions €)', 'Year': 'Election Year'},
            color='Total_Change',
            color_continuous_scale='RdYlBu'
        )
        
        # Add election annotations
        for i, row in election_df.iterrows():
            fig_elections.add_annotation(
                x=row['Year'],
                y=row['Total_Change'],
                text=row['Election'].split(' - ')[0],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                ax=0,
                ay=-40
            )
        
        st.plotly_chart(fig_elections, use_container_width=True)
        
        # Detailed election analysis
        st.subheader("📋 Detailed Election Analysis")
        
        for year, data in election_analysis.items():
            with st.expander(f"🗳️ {year} - {data['election_info']}"):
                st.write(f"**Total Budget Change:** {data['total_change']/1e6:+,.1f} M€")
                
                # Top ministry changes
                sorted_changes = sorted(data['ministry_changes'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                
                st.write("**Top 5 Ministry Changes:**")
                for ministry, change in sorted_changes:
                    color = "🔴" if change < 0 else "🟢"
                    st.write(f"{color} {ministry}: {change/1e6:+.1f} M€")
    
    with tab5:
        st.header("🔮 Future Predictions")
        
        # Perform predictive analysis
        with st.spinner("🤖 Analyzing historical patterns for future predictions..."):
            predictions, historical_data = predict_future_budget(df_filtered, years_ahead=5)
        
        st.subheader("📈 Budget Forecast (Next 5 Years)")
        
        # Create forecast visualization
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=historical_data['Année'],
            y=historical_data['Balance'] / 1e6,
            mode='lines+markers',
            name='Historical Data',
            line=dict(width=4, color='#002395')
        ))
        
        # Add historical events
        for year in historical_data['Année']:
            events = FRENCH_HISTORICAL_EVENTS.get(year, {})
            if 'crisis' in events:
                year_data = historical_data[historical_data['Année'] == year]
                if not year_data.empty:
                    fig_forecast.add_annotation(
                        x=year,
                        y=year_data['Balance'].iloc[0] / 1e6,
                        text=events['crisis'],
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="red",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red"
                    )
        
        # Predictions for each model
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        for i, (model_name, pred_data) in enumerate(predictions.items()):
            fig_forecast.add_trace(go.Scatter(
                x=pred_data['years'],
                y=pred_data['predictions'] / 1e6,
                mode='lines+markers',
                name=f'{model_name} Forecast',
                line=dict(width=3, color=colors[i], dash='dash')
            ))
        
        fig_forecast.update_layout(
            title="🇫🇷 France's Budget Future - Historical Patterns & Predictions",
            xaxis_title="Year",
            yaxis_title="Budget (Millions €)",
            height=600
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Model comparison
        st.subheader("🤖 Model Performance Comparison")
        
        model_comparison = pd.DataFrame({
            'Model': list(predictions.keys()),
            'R² Score': [pred['r2_score'] for pred in predictions.values()]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(model_comparison, use_container_width=True)
        
        with col2:
            fig_models = px.bar(
                model_comparison,
                x='Model',
                y='R² Score',
                title="Model Accuracy Comparison",
                color='R² Score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_models, use_container_width=True)
        
        # Future insights
        st.subheader("🔮 Future Insights")
        
        # Calculate average prediction
        avg_predictions = np.mean([pred['predictions'] for pred in predictions.values()], axis=0)
        future_years = predictions[list(predictions.keys())[0]]['years']
        
        st.write("**Average Predicted Budget Evolution:**")
        for i, year in enumerate(future_years):
            st.write(f"• **{year}:** {avg_predictions[i]/1e6:,.0f} M€")
        
        # Historical context for predictions
        st.write("**Historical Context for Predictions:**")
        recent_trend = historical_data['Balance'].iloc[-1] - historical_data['Balance'].iloc[-3]
        st.write(f"• Recent trend (last 3 years): {recent_trend/1e6:+,.1f} M€")
        
        if recent_trend > 0:
            st.success("📈 **Positive Trend:** Recent budget growth suggests continued positive trajectory")
        else:
            st.warning("📉 **Declining Trend:** Recent budget decline may continue without policy changes")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🇫🇷 France Through Its Budget</h3>
        <p>Understanding France's history and predicting its future through budget data analysis</p>
        <p><strong>Historical Analysis • Crisis Impact • Election Effects • Future Predictions</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
