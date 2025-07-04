
#%%
import os
import streamlit as st
os.environ['LANGSMITH_TRACING']  = st.secrets['LANGSMITH_TRACING']
os.environ['LANGSMITH_ENDPOINT'] = st.secrets['LANGSMITH_ENDPOINT']
os.environ['LANGSMITH_API_KEY']  = st.secrets['LANGSMITH_API_KEY']
os.environ['LANGSMITH_PROJECT']  = st.secrets['LANGSMITH_PROJECT']
os.environ['OPENAI_API_KEY']     = st.secrets['OPENAI_API_KEY']
os.environ['PINECONE_API_KEY']   = st.secrets['PINECONE_API_KEY']

#%%
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI, OpenAIEmbeddings

index_name   = "pinecone-first"
embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

#%%
def retrieve(_dict):
    answer = vector_store.similarity_search(_dict["question"], 1)
    return answer

retrieve({"question":"미국으로 배송되나요?"})

#%%
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm      = ChatOpenAI(model="gpt-4.1-nano", temperature=1.)
parser   = StrOutputParser()
prompt   = PromptTemplate.from_template("""
[Instruction]
당신은 사용자가 자주 묻는 질문에 대하여 정해진 가이드에 따라 답변하는 Faq 가이드입니다.
사용자가 문의한 내용과 그에 대한 답변을 전달받아 답변을 알맞게 다듬어야 합니다.
1. 답변이 문의한 내용에 부합하면, 답변의 주제와 내용이 바뀌지 않도록 주의하면서 거친 단어가 없고 문의자가 반감을 사지 않도록 내용을 조정합니다.
2. 답변이 문의한 내용에 부합하지 않으면, 절대 임의로 말을 지어내지 말고, 담당자의 확인이 필요하므로 기다려야 합니다, 죄송합니다, 라고 하세요.
3. 절대로 답변 외적인 내용을 임의로 덧붙이면 안됩니다.
4. 예시와 상관없이, 답변만 반환해주세요.

[Question & Answer]
Q : {question}
A : {answer}

[Input/Output Example]
[예시1]
Q : 반품 정책이 어떻게 되나요?
A : 제품을 수령한 후 14일 이내에 반품이 가능합니다. 반품 신청은 고객센터에서 도와드립니다.

[예시2]
Q : 배송받은 상자가 뜯어져있는데 보상이 되나요?
A : 담당자와 확인 후 다시 답변드리겠습니다. 불편을 드려 죄송합니다.
""")

chain = ({"question":RunnablePassthrough()}
         | RunnableMap({"question": lambda q: q["question"], "answer": retrieve})
         | prompt | llm | parser)