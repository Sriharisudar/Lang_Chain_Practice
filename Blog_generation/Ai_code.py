import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 1. Modern Embedding & Vector Store Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 2. Modern Core Prompts & Tools
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import Tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 3. Modern Local LLM Integration
from langchain_ollama import OllamaLLM

# 4. Stable Agent Engine
from langchain_classic.agents import create_react_agent, AgentExecutor

# LLM
llm = OllamaLLM(model="gemma3:1b")

# Embeddings
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

Colbert_embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# Load CSVs
csv_files = [
    "credit_card_approval_data_inr.csv",
    "master_credit_card.csv",
    "spending_history_data.csv",
]

docs = []

# Customer Data
rows_per_doc = 50
df = pd.read_csv(csv_files[0])

for i in range(0, len(df), rows_per_doc):
    batch = df.iloc[i:i + rows_per_doc]
    docs.append(
        Document(
            page_content=batch.to_json(orient="records"),
            metadata={
                "source": f"customer_rows_{i+1}_to_{min(i+rows_per_doc, len(df))}"
            },
        )
    )

# Credit Card Master Data
rows_per_doc = 5
df = pd.read_csv(csv_files[1])

for i in range(0, len(df), rows_per_doc):
    batch = df.iloc[i:i + rows_per_doc]
    docs.append(
        Document(
            page_content=batch.to_json(orient="records"),
            metadata={
                "source": f"credit_rows_{i+1}_to_{min(i+rows_per_doc, len(df))}"
            },
        )
    )

# Spending Data
rows_per_doc = 250
df = pd.read_csv(csv_files[2])

for i in range(0, len(df), rows_per_doc):
    batch = df.iloc[i:i + rows_per_doc]
    docs.append(
        Document(
            page_content=batch.to_json(orient="records"),
            metadata={
                "source": f"spending_rows_{i+1}_to_{min(i+rows_per_doc, len(df))}"
            },
        )
    )

# Vector Store
cb_kb = FAISS.from_documents(docs, Colbert_embed)
retriever = cb_kb.as_retriever(search_kwargs={"k": 5})

# RAG Prompt
system_prompt = """
You are a seasoned credit card advisor.

Analyze spending and financial data.
Determine whether the customer is eligible for a credit card.
Suggest the top 3 most suitable credit cards from the provided knowledge base.

Start answer with YES or NO.
Keep response concise.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# RAG Chain (Modern LCEL approach - no langchain.chains needed)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Tool
def credit_advice_tool_func(input_text: str) -> str:
    return rag_chain.invoke(input_text)

credit_advisor_tool = Tool(
    name="CreditAdvisorTool",
    func=credit_advice_tool_func,
    description="Analyzes customer profile and recommends top credit cards.",
)

# ReAct Prompt (manual — no hub dependency)
react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# Agent
agent = create_react_agent(
    llm=llm,
    tools=[credit_advisor_tool],
    prompt=react_prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[credit_advisor_tool],
    verbose=True,
    handle_parsing_errors=True,
)

# Query
query = """
spending habits:
grocery-5000,
dining-3000,
travel-10000

financial history:
income-70000,
cibil score-750,
home loan emi-15000
"""

agent_input = f"""
You are a seasoned credit card advisor capable of analyzing spending patterns and financial history.

Suggest only the top 3 suitable credit cards for this candidate:

{query}

Include:
- Card Name
- Card Type
- Rewards
- Annual Fee
- APR
- Key Benefits

Focus on grocery, dining, travel, and entertainment.
"""

response = agent_executor.invoke(
    {
        "input": agent_input
    }
)
print(response["output"])