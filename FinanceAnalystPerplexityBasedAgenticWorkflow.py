import os
from dotenv import load_dotenv
from openai import OpenAI
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from textblob import TextBlob
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Set up the Perplexity API client
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

# Initialize tools
yfinance_tools = YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)
duckduckgo_tool = DuckDuckGo()

def get_stock_info(symbol):
    stock_price = yfinance_tools.get_current_stock_price(symbol)
    analyst_recommendations = yfinance_tools.get_analyst_recommendations(symbol)
    company_info = yfinance_tools.get_company_info(symbol)
    company_news = yfinance_tools.get_company_news(symbol)
    
    return f"""
    Stock Price: {stock_price}
    Analyst Recommendations: {analyst_recommendations}
    Company Info: {company_info}
    Recent News: {company_news}
    """

def financial_data_agent(symbol):
    stock_info = get_stock_info(symbol)
    messages = [
        {
            "role": "system",
            "content" : f"""You are a financial analysis assistant. Your task is to provide a detailed summary of {symbol} stock predictions from top analysts. Please follow these instructions:
                        1. Create a table with the following columns: Analyst, Firm, Accuracy, Stock, Prediction, and Tentative Date.
                        2. Fill the table with data for at least 5 different analysts, including their name, firm, accuracy percentage, the stock they're analyzing, their prediction (including percentage and direction), and the date by which they expect their prediction to materialize.
                        3. After the first table, create a second table with two columns: Analyst and Accuracy Rating (1-10 scale).
                        4. In this second table, list the same analysts from the first table, but convert their accuracy percentage to a 1-10 scale (e.g., 87% becomes 8.7).
                        5. After both tables, include a brief disclaimer about the nature of these predictions and the importance of personal research and professional advice.

                        Ensure that your response is formatted clearly, with the tables easily readable and the disclaimer separate from the tabular data."""
        },
        {
            "role": "user",
            "content": f"Summarize analyst recommendations for {symbol}. Here's the latest information:\n{stock_info}"
        }
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )
    return response.choices[0].message.content


def web_search_sentiment_agent(symbol):
    search_results = DDGS().text(f"{symbol} stock news latest financial sheets technical indicators", max_results=10)
    
    messages = [
        {
            "role": "system",
            "content": "You are a web search and sentiment analysis AI. Analyze the given search results for the latest news, financial sheets, and technical indicators. Provide a summary and sentiment analysis."
        },
        {
            "role": "user",
            "content": f"Analyze the following search results for {symbol}:\n{search_results}"
        }
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-sonar-huge-128k-online",  # Using a different Perplexity model
        messages=messages,
    )
    analysis = response.choices[0].message.content
    
    sentiment = TextBlob(analysis).sentiment.polarity
    return analysis, sentiment

def main(symbol):
    print(f"Analyzing {symbol}...")
    
    financial_analysis = financial_data_agent(symbol)
    print("\nFinancial Data Analysis:")
    print(financial_analysis)
    
    web_analysis, sentiment = web_search_sentiment_agent(symbol)
    print("\nWeb Search and Sentiment Analysis:")
    print(web_analysis)
    print(f"\nOverall Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")
    print(f"Sentiment Score: {sentiment:.2f}")

if __name__ == "__main__":
    symbol = "PLTR"
    main(symbol)
