import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter as tk
import sys
import os
import pandas as pd
import random
from tkinter import PhotoImage, Label, Entry, Button, StringVar, LabelFrame
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'FNC'))
sys.path.append(r"C:\AI_serv_UI")
sys.path.append(r"C:\AI_serv_UI\FNC")
import FNC.get_info as GIF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


import pandas as pd




entry_vars = {}
entry_widgets = []

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



def update_buttons():
    global idx
    for widget in root.winfo_children():
        widget.destroy()  # 기존 위젯 삭제





def common_data():
    
        global user_data  # 외부 변수를 수정

        # 사용자 입력값 수집
        name = name_entry.get().strip()
        age = age_spinbox.get()
        gender = gender_combobox.get()
        location = location_entry.get().strip()
        
        # 과목 선택 확인
        preferred_subject = [subject for subject, var in zip(subject_list, subject_vars) if var.get()]
        preferred_time = preferred_time_entry.get().strip()

        user_data = {
        'name': name,
        'age': int(age),
        'gender': gender,
        'location': location,
        'preferred_subject': preferred_subject,
        'preferred_time': preferred_time,
        }
        return user_data

def recommend_teachers(user_data):
            

            subject = user_data['preferred_subject']
            preferred_time = user_data['preferred_time']
            location = user_data['location']

            subject = ', '.join(user_data.get('preferred_subject', [])).strip()  # 리스트를 문자열로 변환하고 공백 제거
            preferred_time = str(user_data.get('preferred_time', '')).strip()  # 문자열로 변환 후 공백 제거
            location = str(user_data.get('location', '')).strip()  # 문자열로 변환 후 공백 제거

            tfidf = TfidfVectorizer()
            text_data = teacher_df[['Subject', 'Preferred_time', 'Location']].apply(lambda row: ' '.join(row), axis=1).tolist()
            tfidf_matrix = tfidf.fit_transform(text_data)
            
            stdt_vectors = ' '.join([subject, preferred_time, location]).strip()
            
            if stdt_vectors:
                stdt_vectors = tfidf.transform([stdt_vectors])
                if stdt_vectors.nnz == 0:  # TF-IDF 벡터가 비어있는 경우
                    raise ValueError("학생 데이터가 유효하지 않아 추천을 생성할 수 없습니다.")
            else:
                raise ValueError("학생 데이터가 입력되지 않았습니다.") 
            
            

            sim = cosine_similarity(stdt_vectors, tfidf_matrix).flatten()
            
            top_indices = sim.argsort()[-5:][::-1]
            selected_teachers = teacher_df.iloc[top_indices]
            return selected_teachers

def display_recommendation(selected_teachers):
        root = tk.Tk()
        root.title("과외 선생님 추천 시스템")
        root.geometry("600x700")

        # 추천된 선생님 목록 제목
        Label(root, text="추천된 선생님 목록", font=("Arial", 16)).grid(row=0, column=0, pady=10)

        # 추천된 선생님이 없는 경우 메시지 표시
        if selected_teachers.empty:
            Label(root, text="추천된 선생님이 없습니다.", font=("Arial", 16)).pack(pady=10)
        else:
            # 추천된 선생님 목록 표시
            for index, row in selected_teachers.iterrows():

                frame = LabelFrame(root, text=row['Name'], padx=10, pady=10, relief="solid", bd=1)
                frame.grid(row=index+1, column=0, padx=10, pady=5, sticky="ew")

                # 선생님 정보 표시
                Label(frame, text=f"과목: {row['Subject']}", font=("Arial", 12)).pack(anchor="w")
                Label(frame, text=f"희망 시간: {row['Preferred_time']}", font=("Arial", 12)).pack(anchor="w")
                Label(frame, text=f"지역: {row['Location']}", font=("Arial", 12)).pack(anchor="w")

def on_submit():
    # 학생 정보 입력 후 추천 시스템 실행
    user_data = common_data()  # 사용자 정보 받아오기
    try:
        selected_teachers = recommend_teachers(user_data)  # 추천 선생님 계산
        display_recommendation(selected_teachers)  # 추천 결과 화면에 표시
    except ValueError as e:
        messagebox.showerror("Error", str(e))

user_data = None  # 데이터 저장 변수 초기화
root = tk.Tk()
root.title("학생 정보 입력")

# 이름 입력
tk.Label(root, text="이름:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
name_entry = tk.Entry(root, width=30)
name_entry.grid(row=0, column=1, padx=5, pady=5)

# 나이 선택
tk.Label(root, text="나이 (8세~19세):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
age_spinbox = tk.Spinbox(root, from_=8, to=19, width=5)
age_spinbox.grid(row=1, column=1, padx=5, pady=5)

# 성별 선택
tk.Label(root, text="성별:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
gender_combobox = ttk.Combobox(root, values=["남자", "여자"], state="readonly", width=10)
gender_combobox.grid(row=2, column=1, padx=5, pady=5)
gender_combobox.set("남자")  # 기본값 설정

# 지역 입력
tk.Label(root, text="사는 지역:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
location_entry = tk.Entry(root, width=30)
location_entry.grid(row=3, column=1, padx=5, pady=5)

# 과외 받고 싶은 과목 선택 (여러 개 선택 가능)
tk.Label(root, text="과외 받고 싶은 과목:").grid(row=4, column=0, padx=5, pady=5, sticky="nw")
subject_list = ["국어", "수학", "영어", "사회", "과학"]
subject_vars = [tk.BooleanVar() for _ in subject_list]

# 드롭다운 선택 박스 (OptionMenu)
def on_subject_select(selected_subject):
    for i, subject in enumerate(subject_list):
        if subject == selected_subject:
            subject_vars[i].set(True)  # 선택된 과목만 True로 설정
        else:
            subject_vars[i].set(False)  # 나머지 과목은 False로 설정

# OptionMenu 생성 (기본값은 첫 번째 과목인 '국어')
subject_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=5)
for subject in subject_list:
    subject_listbox.insert(tk.END, subject)

subject_listbox.grid(row=4, column=1, padx=5, pady=5)

# Listbox에서 항목 선택 시 호출될 함수
subject_listbox.bind('<<ListboxSelect>>', on_subject_select)

# 선호하는 과외 시간대 입력
tk.Label(root, text="선호하는 과외 시간대:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
preferred_time_entry = tk.Entry(root, width=30)
preferred_time_entry.grid(row=5, column=1, padx=5, pady=5)

# 제출 버튼
submit_button = tk.Button(root, text="제출", command=on_submit)
submit_button.grid(row=6, column=0, columnspan=2, pady=10)




root.mainloop()
   

    



    

        

    
    



