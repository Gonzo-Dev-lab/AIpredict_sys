import streamlit as st
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'FNC'))
sys.path.append(r"C:\AI_serv_UI")
sys.path.append(r"C:\AI_serv_UI\FNC")
import importlib
import FNC.get_info as GIF
import FNC.utils as ut
importlib.reload(ut)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="선생님 추천", page_icon="🌍")

st.markdown(
    """ 아래에 최고의 선생님들과 함께해 보세요!"""
)





import pandas as pd

# 데이터 가져오기

student_df, unive_list,teacher_df = GIF.get_data()

import random
# List of common Korean family names and given names for random generation
family_names = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임']
given_names = ['민준', '서연', '지훈', '지우', '현우', '수민', '예은', '서윤', '도윤', '채원']

# Replace 'Name' column with random Korean names
teacher_df['Name'] = [random.choice(family_names) + random.choice(given_names) for _ in range(len(teacher_df))]

# Display the first few rows to verify changes
teacher_df.head()


# 데이터 유효성 확인
print("Dataframes received successfully in calling script.")
print("Teacher DataFrame:\n", teacher_df.head())
print("Student DataFrame:\n", student_df.head())
print("University List:", unive_list)




import numpy as np

if not isinstance(teacher_df, pd.DataFrame):
    st.error("teacher_df is not a DataFrame. Please check the data source.")
else:
    teacher_df = teacher_df[['Name', 'Subject', 'Preferred_time', 'Location']]

# user_data 로딩
if not hasattr(ut, 'common_data'):
    raise AttributeError("utils 모듈에 common_data 함수가 정의되어 있지 않습니다.")
user_data= ut.common_data(unive_list)


def Recomandation (teacher_df, user_data):
        
        #코사인 유사도 측정, 추천하는 시스템 구현

        

        subject = user_data['preferred_subject']
        preferred_time = user_data['preferred_time']
        location = user_data['location']

        teacher_df = pd.DataFrame(teacher_df)

        tfidf =TfidfVectorizer()

        #선생님 데이터 벡터화
        text_data = teacher_df[['Subject', 'Preferred_time', 'Location']].apply(lambda row: ' '.join(row), axis=1).tolist()
        tfidf_matrix = tfidf.fit_transform(text_data) #TF-IDF 벡터화
        

        #벡터화
        subject = ', '.join(user_data.get('preferred_subject', [])).strip()  # 리스트를 문자열로 변환하고 공백 제거
        preferred_time = str(user_data.get('preferred_time', '')).strip()  # 문자열로 변환 후 공백 제거
        location = str(user_data.get('location', '')).strip()  # 문자열로 변환 후 공백 제거

        # 세 문자열을 하나의 텍스트로 결합
        stdt_vectors = ' '.join([subject, preferred_time, location]).strip() 


        # Join the strings into one
        print(f"Student vector: {stdt_vectors}")

        if stdt_vectors:  
            stdt_vectors = tfidf.transform([stdt_vectors])  
        if stdt_vectors.nnz == 0:
             print("Error: Empty or invalid input for TF-IDF.")
             return 
        

        print('학생 벡터 :', stdt_vectors)

       



        print(stdt_vectors.shape)
        print(tfidf_matrix.shape)

        sim = cosine_similarity(stdt_vectors, tfidf_matrix).flatten()

        print(teacher_df.columns)


        top_indices = sim.argsort()[-5:][::-1]
        top_teachers = teacher_df.iloc[top_indices][['Name', 'Subject', 'Preferred_time', 'Location']]

        st.title("과외 선생님 매칭!")

        for _, row in top_teachers.iterrows():
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h4>{row['Name']}</h4>
                    <p><strong>Subject:</strong> {row['Subject']}</p>
                    <p><strong>Preferred Time:</strong> {row['Preferred_time']}</p>
                    <p><strong>Location:</strong> {row['Location']}</p>
                    <p><strong>Similarity:</strong> {sim[_]:.2f}</p>
                </div>
                """, unsafe_allow_html=True)


# user_data가 비어있지 않다면 추천 시스템 실행
if st.button("선생님 추천 받기"):
    if user_data:  
        Recomandation(teacher_df, user_data)
    else:
        st.warning("데이터를 입력한 후에 추천을 받을 수 있습니다.")

   

    



    

        

    
    



