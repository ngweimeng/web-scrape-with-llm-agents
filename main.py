import csv
import json
import os
from dotenv import load_dotenv
load_dotenv()
import requests
from bs4 import BeautifulSoup
import box
import pandas as pd
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from src.agents import create_agent_executor
from src.llm import llm
from src.prompts import system_prompt, generate_input_prompt
from src.utils import default_values

# Load configuration
def load_config():
    with open("config/config.yaml", "r", encoding="utf8") as ymlfile:
        return box.Box(yaml.safe_load(ymlfile))

cfg = load_config()

# --- athome.lu scraper ---
def scrape_athome(location: str, min_rent: int = None, max_rent: int = None) -> list:
    """
    Query athome.lu rental listings for a given location and optional rent range.
    Returns a list of dicts: {title, price, address, link}
    """
    base_url = "https://www.athome.lu/louer"
    params = {'transactionTypes': 'Louer', 'localite': location}
    if min_rent:
        params['rent_min'] = min_rent
    if max_rent:
        params['rent_max'] = max_rent

    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for card in soup.select('.adCard'):
        title = card.select_one('.adCard-title')
        price = card.select_one('.adCard-price')
        addr = card.select_one('.adCard-locality')
        link_el = card.select_one('a.adCard-link')
        results.append({
            'title': title.get_text(strip=True) if title else None,
            'price': price.get_text(strip=True) if price else None,
            'address': addr.get_text(strip=True) if addr else None,
            'link': f"https://www.athome.lu{link_el['href']}" if link_el and link_el.has_attr('href') else None,
        })
    return results

# Wrap scraper as a LangChain Tool
athome_tool = Tool(
    name="athome_scraper",
    func=scrape_athome,
    description="Use this tool to fetch rental listings from athome.lu. Input should be a JSON string with keys 'location', 'min_rent', and 'max_rent'."
)

# Bind tool to OpenAI function schema
tools = [athome_tool]
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Build agent executor
agent_executor = create_agent_executor(
    prompt=prompt,
    llm_with_tools=llm_with_tools,
    tools=tools
)

# Retry-decorated executor function
@retry(stop=stop_after_attempt(2), wait=wait_fixed(10), retry_error_callback=default_values)
def execute_housing_agent(
    query: str,
):
    """
    Execute the LangChain agent to fetch athome.lu listings based on a natural-language query.
    Query should specify location, and optionally rent range.
    Returns the tool output parsed as JSON.
    """
    # Agent invocation
    with get_openai_callback() as cb:
        response = agent_executor.invoke({"input": query})
        cost = cb.total_cost
        tokens = cb.total_tokens

    # The agent returns JSON string if it used the function
    output_str = response.get("output") or response.get("function_response")
    try:
        data = json.loads(output_str)
    except ValueError:
        # If plain text, wrap into message
        data = {"message": output_str}

    # Attach metadata
    data["llm_cost"] = cost
    data["llm_tokens"] = tokens
    return data

if __name__ == '__main__':
    # Example usage
    query = "Find rental listings in Luxembourg, Limpertsberg with rent between 1000 and 1500 euros"
    print(json.dumps(execute_housing_agent(query), indent=2, ensure_ascii=False))
