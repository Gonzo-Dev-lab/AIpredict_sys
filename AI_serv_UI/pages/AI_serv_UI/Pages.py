import streamlit as st

st.write('#somethingname에 온걸 환영합니다.')

st.sidebar.success("Select a pages above.")

st.markdown(
    """
    이 프로그램은 여러분의 성적을 받아, 대학 합격률을 예측합니다.
    또한 여러분이 배우고 싶은 과목의 최고의 선생님을 5명까지 추천해드립니다.
    시간과 돈을 절약하세요.
"""
)