import streamlit as st

def common_data(unive_list):
    st.header("학생 정보 입력")
        

    # 사용자 정보 입력 받기
    name = st.text_input("이름을 입력하세요", key='name_input')
    age = st.number_input("나이를 입력하세요(8세~19세)", min_value=8, max_value=19)
    gender = st.selectbox("성별을 선택하세요", ["남자", "여자"])
                
    

        # 과외 경험 및 지역 정보 입력 받기
    
    location = st.text_input("사는 지역을 입력하세요 특별시/광역시/도, 시/군/구, 동/읍/면", key = 'location_input')
        
    # 과외 받고 싶은 과목 선택 (여러 개 선택 가능)
    subject_list = ["국어", "수학", "영어", "사회", "과학"]
    preferred_subject = st.multiselect("과외 받고 싶은 과목을 선택하세요 (여러 개 선택가능)", subject_list)

        # 선호하는 과외 시간대 입력 받기
    preferred_time = st.text_input("선호하는 과외 시간대를 입력하세요 (예: 주말 오후, 평일 저녁 등)", key = 'time_input')
            
                
    
    user_data = {
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'location': location,
                    'preferred_subject': preferred_subject,
                    'preferred_time': preferred_time,
                }
    return user_data
