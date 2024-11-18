import streamlit as st
st.set_page_config(page_title="학생 정보 입력 및 예측", page_icon="🌍")
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'FNC'))
sys.path.append(r"C:\Users\wawa2\OneDrive\바탕 화면\AI_serv_UI")
sys.path.append(r"C:\Users\wawa2\OneDrive\바탕 화면\AI_serv_UI\FNC")
import FNC.get_info as GIF
import FNC.utils as ut
from sklearn.impute import KNNImputer





current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

university_rankings = {
    "Seoul National University": 50,
    "KAIST": 49,
    "Yonsei University": 48,
    "POSTECH": 47,
    "Korea University": 46,
    "Sungkyunkwan University": 45,
    "Hanyang University": 44,
    "Kyung Hee University": 43,
    "UNIST": 42,
    "Sogang University": 41,
    "GIST": 40,
    "Ewha Womans University": 39,
    "Chung-Ang University": 38,
    "Ulsan University": 37,
    "Ajou University": 36,
    "Pusan National University": 35,
    "Konkuk University": 34,
    "Inha University": 33,
    "Kyungpook National University": 32,
    "Sejong University": 31,
    "Yeungnam University": 30,
    "Jeonbuk National University": 29,
    "Chonnam National University": 28,
    "Chungnam National University": 27,
    "University of Seoul": 26,
    "Hallym University": 25,
    "Dongguk University": 24,
    "Gangwon National University": 23,
    "Catholic University": 22,
    "Chungbuk National University": 21,
    "Hankuk University of Foreign Studies": 20,
    "Calvin University": 19,
    "Jeju National University of Education": 18,
    "Seoul Tech": 17,
    "Kookmin University": 16,
    "Gyeongsang National University": 15,
    "Incheon University": 14,
    "Sungshin Women's University": 13,
    "Dankook University": 12,
    "Sunchon National University": 11,
    "Pukyong National University": 10,
    "Sookmyung Women's University": 9,
    "Catholic University of Korea": 8,
    "Inje University": 7,
    "Kyonggi University": 6,
    "Hongik University": 5,
    "DGIST": 4,
    "KUST": 3
        }


@st.cache_data
def load_and_preprocess_data():
    stdf, _, _= GIF.get_data()    
    
    # 대학 순위를 student.df에 추가
    stdf['University_rank'] = stdf['University'].map(university_rankings)
    
    # 'serial No' 열 제거
    if 'serial No' in stdf.columns:
        stdf = stdf.drop(columns=['serial No'])
    
    stdf = stdf.drop(columns=['University'])

    # 숫자형 데이터만 선택 및 결측치 처리
    stdf_numeric = stdf.select_dtypes(include=[np.number])
    if stdf_numeric.isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        stdf_imputed = imputer.fit_transform(stdf_numeric)
        stdf_imputed_df = pd.DataFrame(stdf_imputed, columns=stdf_numeric.columns)
        stdf.update(stdf_imputed_df)
    
    # 이상치 처리
    lower_bound = stdf.quantile(0.01)
    upper_bound = stdf.quantile(0.99)
    stdf = stdf.clip(lower=lower_bound, upper=upper_bound, axis=1)
    
    return stdf

@st.cache_resource
def create_and_train_model(x_data, y_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    
    model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])
    
    
    model.fit(x_data, y_data, epochs=150)
    
    return model




stdf = load_and_preprocess_data()

x_data = []
y_data = stdf['chance of admit'].values

for i, row in stdf.iterrows():
                    x_data.append([
                        row['Korean'], row['Math'], row['English'], 
                        row['Social Studies'], row['Science'], 
                        row['Tutoring Period'], row['University_rank']
                    ])
                                
x_data = pd.DataFrame(x_data, columns=['Korean', 'Math', 'English', 'Social Studies', 'Science', 'Tutoring Period', 'University_rank'])
                
                            
x_data = np.array(x_data)
y_data = np.array(y_data)
                            
                        
scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)
        
    
model = create_and_train_model(x_data, y_data)

_, unive_list, _ = GIF.get_data()


korean_score = st.number_input("국어 성적을 입력하세요", min_value=0, max_value=100)
math_score = st.number_input("수학 성적을 입력하세요", min_value=0, max_value=100)
english_score = st.number_input("영어 성적을 입력하세요", min_value=0, max_value=100)
social_score = st.number_input("사회 성적을 입력하세요", min_value=0, max_value=100)
science_score = st.number_input("과학 성적을 입력하세요", min_value=0, max_value=100)

scores = [korean_score, math_score, english_score, social_score, science_score]
tutoring_experience = st.number_input("과외 받은 기간(1~4년)을 입력하세요", min_value=0, max_value = 4)

target_univ = st.selectbox('목표 대학을 정하십시오', unive_list)



def main(model):
    

    st.header("합격률 예측")


    if target_univ is not None:
        st.write(f"목표대학 {target_univ}의 합격률을 예측하시겠습니까?")

                
        if st.button("예", key="yes_button"):
                        
            with st.spinner("잠시만 기다려 주십시오..."):
                            
                # 예측
                target_univ_rank = university_rankings.get(target_univ, 0)
                input_data = np.array(scores + [tutoring_experience] + [target_univ_rank])
                input_data = input_data.reshape(1, -1)
                input_data = scaler.transform(input_data)
                res = model.predict(input_data)
                            
                st.write(f"{target_univ}의 합격률은 {res[0][0] * 100:.2f}% 입니다.")
    else:
        st.write("먼저 학생 정보를 입력해 주세요.")

if __name__ == "__main__":
    main(model)

    

    
    



   
    
    
    
    



    
    

    
    
    
    
    
    
   
   

            
            
   