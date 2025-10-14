# 🇫🇷 France Through Its Budget - Historical Storytelling Dashboard

## 🎯 Project Vision
This application tells the story of France from 2012-2022 through its budget data, connecting financial decisions to historical events, elections, and national crises. It's designed to understand France's past and predict its future through data analysis.

## 📖 What This Application Does

### 🏛️ Historical Storytelling
- **Connects budget data to real events**: See how terrorist attacks (2015), COVID-19 (2020), and elections impacted spending
- **Tells France's story**: Understand the country's evolution through financial decisions
- **Crisis analysis**: Detailed breakdown of how crises affected different ministries
- **Election impact**: Analyze how presidential elections changed budget priorities

### 🔮 Future Predictions
- **Machine Learning forecasts**: Predict France's budget trajectory for the next 5 years
- **Historical patterns**: Use past events to understand future trends
- **Multiple models**: Compare different prediction algorithms
- **Context-aware predictions**: Consider historical events in forecasts

## 🚀 Key Features

### 📊 Historical Analysis
- **Timeline visualization**: See France's budget evolution with event markers
- **Crisis impact analysis**: Detailed breakdown of 2015 attacks, COVID-19, Ukraine war
- **Election correlation**: How presidential elections affected budget priorities
- **Ministry performance**: Track how different ministries evolved over time

### 🎨 Storytelling Elements
- **Event cards**: Detailed information about key historical events
- **Crisis indicators**: Visual markers for major crises and their budget impact
- **Election markers**: Highlight election years and their financial consequences
- **Context explanations**: Understand why budget changes happened

### 🔮 Predictive Analytics
- **5-year forecasts**: Predict France's budget trajectory
- **Model comparison**: Compare Linear Regression, Ridge, and Random Forest
- **Historical context**: Use past events to inform future predictions
- **Trend analysis**: Understand recent patterns and their implications

## 📋 Application Structure

### 🏠 Main Dashboard
- **Key metrics**: Total budget, crisis years, election years
- **Historical timeline**: Visual story of France's budget evolution
- **Event summaries**: Detailed breakdown of each year's events

### 📊 Analysis Tabs

#### 1. 📖 Historical Story
- **Comprehensive timeline**: France's budget evolution with event markers
- **Event details**: Expandable cards for each year's key events
- **Budget impact**: How events affected spending decisions

#### 2. 🏛️ Ministry Analysis
- **Top ministries**: Ranked by total budget over the period
- **Evolution tracking**: See how ministries changed over time
- **Performance metrics**: Detailed statistics for each ministry
- **Removed confusing charts**: Cleaned up interface based on feedback

#### 3. 🚨 Crisis Impact Analysis
- **Crisis selection**: Choose specific crisis years (2015, 2020, 2022)
- **Ministry breakdown**: See which ministries were most affected
- **Change analysis**: Quantify the impact of each crisis
- **Visual indicators**: Clear crisis impact visualization

#### 4. 🗳️ Election Analysis
- **Election years**: 2012, 2017, 2022 presidential elections
- **Budget changes**: How elections affected spending priorities
- **Ministry impacts**: Which ministries gained/lost funding
- **Political context**: Connect budget changes to political shifts

#### 5. 🔮 Future Predictions
- **5-year forecast**: Predict France's budget trajectory
- **Model comparison**: Compare different ML algorithms
- **Historical context**: Use past patterns for predictions
- **Future insights**: Interpret what predictions mean for France

## 🎯 Historical Events Covered

### 🚨 Major Crises
- **2015**: Terrorist attacks (Charlie Hebdo, Bataclan) → Defense spending explosion
- **2020**: COVID-19 pandemic → Massive health spending
- **2022**: War in Ukraine → Defense and energy spending

### 🗳️ Elections
- **2012**: Hollande elected → Social spending increases
- **2017**: Macron elected → Economic reforms, digital transformation
- **2022**: Macron re-elected → Continued reforms, crisis management

### 📈 Key Trends
- **Defense spending**: Increased after 2015 attacks
- **Health spending**: Massive increase during COVID-19
- **Social spending**: Fluctuated with political changes
- **Digital transformation**: Accelerated under Macron

## 🔧 Technical Features

### 📊 Data Processing
- **Historical context**: Events mapped to budget data
- **Crisis detection**: Automatic identification of crisis years
- **Election tracking**: Presidential election year analysis
- **Ministry categorization**: Organized by government departments

### 🤖 Machine Learning
- **Multiple models**: Linear Regression, Ridge, Random Forest
- **Historical training**: Uses 2012-2022 data for predictions
- **Performance metrics**: R² scores and accuracy measures
- **Context awareness**: Considers historical events in predictions

### 🎨 Visualization
- **Interactive charts**: Plotly-based dynamic visualizations
- **Event markers**: Visual indicators for crises and elections
- **Color coding**: Intuitive representation of different data types
- **Responsive design**: Works on all devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Budget data files (generated by Jupyter notebook)

### Installation
```bash
# Clone the repository
git clone <your-repo>
cd Data_viz_20220797

# Install dependencies
pip install -r requirements.txt

# Run the historical storytelling application
python run_app.py
```

### Alternative Launch
```bash
# Direct Streamlit launch
streamlit run app_historical.py
```

## 📈 Key Insights You'll Discover

### 🚨 Crisis Impact
- **2015 Attacks**: Defense spending increased by X% after terrorist attacks
- **COVID-19**: Health spending exploded during pandemic
- **Ukraine War**: Defense and energy spending surged in 2022

### 🗳️ Election Effects
- **2012**: Hollande's election led to increased social spending
- **2017**: Macron's election brought digital transformation focus
- **2022**: Re-election maintained reform momentum

### 🔮 Future Trends
- **Budget trajectory**: Where France's budget is heading
- **Ministry priorities**: Which areas will receive more/less funding
- **Economic implications**: What budget changes mean for France

## 🎨 Design Philosophy

### 📖 Storytelling Approach
- **Narrative structure**: Each analysis tells a story
- **Historical context**: Events explained with budget data
- **Visual storytelling**: Charts that tell France's story
- **Educational value**: Learn French history through data

### 🎯 User Experience
- **Intuitive navigation**: Clear tab structure
- **Contextual information**: Events explained with data
- **Visual hierarchy**: Important information highlighted
- **Interactive elements**: Explore data at your own pace

## 🔮 Future Enhancements

### 📊 Additional Analysis
- **Regional analysis**: Budget impact by French regions
- **International comparison**: Compare with other EU countries
- **Sector analysis**: Deep dive into specific economic sectors
- **Policy impact**: Measure effectiveness of government policies

### 🤖 Advanced ML
- **Time series models**: More sophisticated forecasting
- **Crisis prediction**: Predict future crisis impacts
- **Policy simulation**: Model effects of policy changes
- **Scenario analysis**: Multiple future scenarios

## 📚 Educational Value

### 🎓 Learning Outcomes
- **French history**: Understand recent French history through data
- **Budget analysis**: Learn how government budgets work
- **Data science**: See ML applied to real-world problems
- **Policy analysis**: Understand how policies affect budgets

### 📖 Use Cases
- **Students**: Learn about French politics and economics
- **Researchers**: Analyze government spending patterns
- **Policy makers**: Understand budget impact of decisions
- **Citizens**: Better understand government spending

## 🛠️ Technical Architecture

### 📊 Data Flow
1. **Load data**: Clean budget data with historical context
2. **Process events**: Map historical events to budget years
3. **Analyze patterns**: Identify trends and correlations
4. **Generate insights**: Create automated recommendations
5. **Predict future**: Use ML to forecast trends

### 🔧 Technology Stack
- **Frontend**: Streamlit with custom CSS
- **Visualization**: Plotly for interactive charts
- **ML**: Scikit-learn for predictions
- **Data**: Pandas for processing
- **Analysis**: SciPy for statistics

## 📞 Support & Documentation

### 📚 Resources
- **Code comments**: Detailed explanations throughout
- **Function documentation**: Comprehensive docstrings
- **User guide**: Step-by-step instructions
- **Historical context**: Background on French events

### 🤝 Contributing
- **Feedback welcome**: Suggestions for improvements
- **Bug reports**: Help identify issues
- **Feature requests**: Suggest new analyses
- **Historical accuracy**: Help improve event data

---

**🇫🇷 Built to tell France's story through data**

*Understanding France's past and predicting its future through budget analysis*
