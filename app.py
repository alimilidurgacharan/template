from flask import Flask, render_template, request, jsonify
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
import markdown
import re
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

groq_api = os.getenv('GROQ_API_KEY')

app = Flask(__name__)

# Define Stock Analysis Agent
stock_analysis_agent = Agent(
    name='Stock Analysis Agent',
    model=Groq(id="deepseek-r1-distill-llama-70b", api_key=groq_api),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True),
        DuckDuckGoTools()
    ],
    instructions=[
        "Use DuckDuckGoTools for real-time stock-related news.",
        "Use YFinanceTools for stock prices, company details, and analyst recommendations.",
        "Provide top 3 recent news headlines with short summaries.",
    ],
    show_tool_calls=False,
    markdown=True
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker').strip().upper()
    temperature = float(request.form.get('temperature', 0.2))
    if not ticker:
        return jsonify({'error': 'Please enter a valid stock ticker.'})
    
    try:
        stock_data = yf.Ticker(ticker)
        stock_info = stock_data.info

        # Fetch real-time stock price
        current_price = stock_info.get('regularMarketPrice', 'N/A')
        after_hours_price = stock_info.get('postMarketPrice', 'N/A')
        previous_close = stock_info.get('previousClose', 'N/A')

        # Handle missing real-time price
        if current_price == 'N/A':
            current_price = previous_close

        # Construct stock price message
        price_message = f"📊 **Live Stock Price**: ${current_price:.2f}" if current_price != 'N/A' else "📊 **Stock Price Unavailable**"
        if after_hours_price and after_hours_price != 'N/A':
            price_message += f"\n📉 **After-Hours Price**: ${after_hours_price:.2f}"

        structured_prompt = f"""
        You are a **Stock Analysis AI**. Your job is to analyze **{ticker}**.

        📌 **Instructions**:
        - Use **YFinanceTools** to get:
        - **Real-time stock price**
        - **Company details (market cap, sector, key financials)**
        - **Analyst recommendations**
        - **Latest stock news**
        - **Technical analysis & trends**
        - **Do NOT use DuckDuckGo for stock prices**.
        - **If real-time price is missing, use the last closing price with a note**.

        📊 **Stock Price**:  
        {price_message}  

        📈 **Response Format (Json)**:
        1. **Company Overview**
        2. **Stock Performance**
        3. **Recent News**
        4. **Analyst Ratings**
        5. **Technical Trend Analysis**
        6. **Final Buy/Hold/Sell Recommendation**

        Strictly Return your response in **clean Json format**.

        Expected json format::
        {{
            "Company Overview": "",
            "Stock Performance": "",
            "Recent News": "",
            "Analyst Ratings": "",
            "Technical Trend Analysis": "",
            "Final Buy/Hold/Sell Recommendation": "",
        }}

        """


        # structured_prompt=f"""
        # You are a specialized Stock Analysis Assistant designed to provide comprehensive insights about {ticker}.

        # ## Data Collection Instructions

        # Retrieve the following information using YFinanceTools:
        # - Current stock price and trading volume
        # - Historical price data (1-day, 1-week, 1-month, YTD, 1-year)
        # - Company fundamentals (market cap, P/E ratio, EPS, dividend yield, sector, industry)
        # - Recent quarterly/annual financial metrics (revenue, profit margins, debt-to-equity)
        # - Analyst recommendations and price targets
        # - Latest news articles affecting the stock (past 7 days)
        # - Technical indicators (50/200-day MA, RSI, MACD, Bollinger Bands)

        # Important: Do NOT use DuckDuckGo for stock price data. If real-time price is unavailable, use the latest closing price and clearly indicate this.

        # ## Current Stock Information
        # {price_message}

        # ## Analysis Requirements

        # Provide a thorough analysis that includes:
        # 1. Contextual interpretation of all metrics relative to sector averages and historical performance
        # 2. Identification of key price movement catalysts
        # 3. Assessment of both bullish and bearish factors
        # 4. Evaluation of risk factors specific to this company
        # 5. Clear reasoning behind your final recommendation

        # ## Response Format

        # Strictly Return your analysis in this exact JSON structure:


        # {{
        # "Company Overview": "Comprehensive company profile including business model, competitive position, market share, and key financial metrics",
        
        # "Stock Performance": "Detailed price analysis including YTD return, comparison to relevant indices, price volatility, and notable price movements",
        
        # "Recent News": "Summary of 3-5 most significant recent developments affecting stock price with their market impact",
        
        # "Analyst Ratings": "Current consensus rating, price targets (low/average/high), and notable analyst opinion changes in past month",
        
        # "Technical Trend Analysis": "Interpretation of key technical indicators, support/resistance levels, and pattern identification",
        
        # "Final Buy/Hold/Sell Recommendation": "Clear investment recommendation with specific justification and risk assessment"
        # }}
        


        response = stock_analysis_agent.run(structured_prompt, temperature=temperature )
        analysis_content = response.content if hasattr(response, "content") else "No analysis available."
        analysis_content = re.sub(r"<\|tool.*?\|>", "", analysis_content).strip()

        # Fetch stock history for plots
        history = stock_data.history(period="3mo")
        if history.empty:
            return jsonify({'error': f'No stock data found for {ticker}.'})

        plot_data = {
            'dates': history.index.strftime('%Y-%m-%d').tolist(),
            'open': history['Open'].tolist(),
            'high': history['High'].tolist(),
            'low': history['Low'].tolist(),
            'close': history['Close'].tolist(),
            'volume': history['Volume'].tolist()
        }

        disclaimer = (
            "<div class='alert alert-warning'>Disclaimer: The stock analysis and recommendations provided "
            "are for informational purposes only and should not be considered financial advice. Always do your "
            "own research or consult with a financial professional.</div>"
        )

        markdown_content = markdown.markdown(analysis_content)
        cleaned_content = re.sub(r'```.*?```', '', markdown_content, flags=re.DOTALL)
        formatted_response = re.sub(r'<table>', '<table class="table table-bordered table-striped">', cleaned_content)
        print(formatted_response)

        return jsonify({
            'result': formatted_response,
            'plot_data': plot_data,
            'disclaimer': disclaimer
        })

    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'})

if __name__ == "__main__":
    app.run(debug = True)
