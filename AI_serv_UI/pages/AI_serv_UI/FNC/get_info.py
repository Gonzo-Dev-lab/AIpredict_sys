import pandas as pd
import os




def get_data() :
    # 현재 스크립트의 디렉토리 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'student_data.csv')

    # CSV 파일 읽기
    student_df = pd.read_csv(csv_path)
    teacher_df = pd.read_csv(r"C:\AI_serv_UI\FNC\teacher_data.csv")

    

    univ_list = student_df['University'].unique().tolist()
    
    
    return student_df, univ_list, teacher_df

get_data()
