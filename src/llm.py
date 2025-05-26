# Exposes an llm instance with .bind() for functions
from langchain.chat_models import ChatOpenAI

# Adjust model name and parameters as needed
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)