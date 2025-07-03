import streamlit as st
from homework_fag_chatbot_rag_20250713 import *

st.set_page_config(page_title="Q&A Chat", layout="centered")

st.title("💬 FAQ 챗봇")
st.caption("Streamlit + OpenAI + 질문 기록")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("chat_form", clear_on_submit=True):
    question  = st.text_input("질문을 입력하세요:", "")
    submitted = st.form_submit_button("질문하기")

if submitted and question:
    with st.spinner("답변 생성 중..."):
        answer = chain.invoke(question)

    # 기록 저장
    st.session_state.chat_history.append({"question": question, "answer": answer})

# 히스토리 출력
if st.session_state.chat_history:
    st.subheader("📜 이전 대화 기록")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q{i+1}: {chat['question']}"):
            st.markdown(f"**A:** {chat['answer']}")
