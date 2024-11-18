import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import get_info as GIN






def get_university_rank(target):
    
    student_df,_= GIN.get_data()
    # 타겟 대학의 랭크 추출
    try:
        rank = student_df.loc[student_df['University'] == target, 'University_rank'].values[0]
        return rank
        
    except IndexError:
        return "대학이 데이터에 없습니다."