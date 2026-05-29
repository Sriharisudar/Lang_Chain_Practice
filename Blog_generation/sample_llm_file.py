# from langchain.llms import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
llm = Ollama(model="qwen3:4b")
# Recommended HuggingFace embedding model for LangChain
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

Colbert_embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# === Load and Combine CSVs ===
csv_files = [
    'credit_card_approval_data_inr.csv',
    'master_credit_card.csv',
    'spending_history_data.csv'
]

docs, rows_per_doc = [], 50
df = pd.read_csv(csv_files[0])
# print(f"Loaded {len(df)} rows from {file}")
for i in range(0, len(df), rows_per_doc):
    batch = df.iloc[i:i + rows_per_doc]
    doc_text = batch.to_json(orient='records')
    document = Document(
        page_content=doc_text,
        metadata={"source": f"customer_rows_{i + 1}_to_{min(i + rows_per_doc, len(df))}"}
    )
    docs.append(document)

rows_per_doc = 5
df = pd.read_csv(csv_files[1])
for i in range(0, len(df), rows_per_doc):
    batch = df.iloc[i:i + rows_per_doc]
    doc_text = batch.to_json(orient='records')
    document = Document(
        page_content=doc_text,
        metadata={"source": f"credit_rows_{i + 1}_to_{min(i + rows_per_doc, len(df))}"}
    )
    docs.append(document)

rows_per_doc = 250
df = pd.read_csv(csv_files[2])
# print(f"Loaded {len(df)} rows from {file}")
for i in range(0, len(df), rows_per_doc):
    batch = df.iloc[i:i + rows_per_doc]
    doc_text = batch.to_json(orient='records')
    document = Document(
        page_content=doc_text,
        metadata={"source": f"spending_rows_{i + 1}_to_{min(i + rows_per_doc, len(df))}"}
    )
    docs.append(document)

cb_kb = FAISS.from_documents(docs, Colbert_embed)
colbert_retriever = cb_kb.as_retriever()

system_prompt = (
    "you are seasoned credit card advisor. by analysing the spening and financial data"
    "based on the data provided tell given person is given credit card or not "
    "suggest top 3 credit card suitable for candidate in given knowledge base"
    "start your answer with yes or no. Use one liner maximum and keep the answer concise. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(colbert_retriever, question_answer_chain)


def credit_advice_tool_func(input: str) -> str:
    result = rag_chain.invoke({"input": input})['answer']
    return result.strip()


credit_advisor_tool = Tool(
    name="CreditAdvisorTool",
    func=credit_advice_tool_func,
    description="Provides credit card names and its benefits."
)

agent = initialize_agent(
    tools=[credit_advisor_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

query = "spending habits for grocery-5000, dining-3000, travel-10000  and financial history income 70000, cibil score-750, home loan emi 150000"

agent_input = (
    f"""you are seasoned credit card advisor. capable of analysing the spending patterns and financial history
    your job is to suggest only top 3 suitable credit card for candidate has : {query} 
    include card type, rewards, annual fee, APR and benefits focusing on grocery, dining, travel, entertainment."""
)

respose_from_agent = agent.run(agent_input)
print(respose_from_agent)