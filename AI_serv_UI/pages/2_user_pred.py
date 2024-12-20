import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'FNC'))
sys.path.append(r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI")
sys.path.append(r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\FNC")
import FNC.get_info as GIF
import pages.utils as ut
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



def load_and_preprocess_data():
    stdf, _, _= GIF.get_data()    
    

    # 대학 순위를 student.df에 추가
    stdf['University_rank'] = stdf['University'].map(university_rankings)
    
    
    #랭크가 0인 대학들을 대학랭크들의 평균값으로 계싼
    avg_rank = stdf.loc[stdf['University_rank'] > 0, 'University_rank'].mean()  # 0 제외한 평균 계산
    stdf['University_rank'] = stdf['University_rank'].replace(0, avg_rank)
    
    
    
    
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
    
    
    
    return stdf

stdf = load_and_preprocess_data()



def create_and_train_model(x_data, y_data):
    
 
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

        
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  loss='mse', metrics=['accuracy'])
    
    
    
    model.fit(x_data, y_data, epochs=100)
    

    
    return model
    
 


x_data = []
y_data = stdf['chance of admit'].values

for i, row in stdf.iterrows():
                    x_data.append([
                        row['Korean'], row['Math'], row['English'], 
                        row['Inquiry 1'], row['Inquiry 2'], 
                        row['Tutoring Period'], row['University_rank']
                    ])
                                
x_data = pd.DataFrame(x_data, columns=['Korean', 'Math', 'English', 'Inquiry 1', 'Inquiry 2', 'Tutoring Period', 'University_rank'])

                
                            
x_data = np.array(x_data)
y_data = np.array(y_data)


print(x_data) 
                        
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)


negative_values = x_data[x_data < 0]
print("Number of negative values:", negative_values.size)
print("Locations of negative values:", np.where(x_data < 0))
            


model = create_and_train_model(x_data, y_data)


from tkinter import *
from tkinter import ttk




root = Tk()


windows = root.geometry("550x650+150+150")

height = 400
width = 350

idx = 0


CheckVar1 = IntVar(value=0)

entry_widgets = []  # Entry 위젯 리스트
entry_labels = []  # 라벨 위젯 리스트
entry_vars = []

def clear_entries():
    """Entry 위젯과 라벨을 숨깁니다."""
    for entry in entry_widgets:
        entry.pack_forget()  # Entry 숨기기
    for label in entry_labels:
        label.pack_forget() 
    target_univ_combobox.place_forget

def nextpage() :
    global idx

    if idx == 1 and CheckVar1.get() == 0:
        return 

    idx += 1
    labelimg.config(image = imges[idx])   
    update_buttons() 

    if idx == 2:  # 2번째 페이지에서 엔트리와 라벨 표시
        # 이전 페이지의 엔트리와 라벨을 숨깁니다.
        update_buttons() 
      
    if idx == 3:
        clear_entries()
        update_buttons()

    if idx == 4:
        update_buttons()



def prepage():
    global idx

    idx -= 1
    labelimg.config(image = imges[idx])
    if idx == 4:
        update_buttons()

    if idx == 3:
        update_buttons()

    if idx ==  2 :
        update_buttons() 
       
    if idx == 1 :
        clear_entries()

def plusidx():
    global idx
    idx += 1
    labelimg.config(image = imges[idx])  


def update_buttons():

    prebtn.config(state=NORMAL if idx > 0 else DISABLED)

    nextbtn.config(state=NORMAL if idx < len(imges) - 1 else DISABLED)


    if idx == 1:  # 개인정보 동의 이미지
        agreebtn.place(x=350, y=520)
        
    else :
        agreebtn.place_forget()

    if idx == 2 : 
        
        labelimg.pack_forget()
        for i, entry in enumerate(entry_widgets) :
            entry_labels[i].pack()
            entry.pack()
        
        yesButton.config(state=NORMAL, command = task )
                
        
        target_univ_label.pack() 
        target_univ_combobox.pack() # 대학 설정 박스
        predictlabel.place(x= 125, y = 400)
        show_values_btn.pack()
        yesButton.pack() # 예측 트리거 버튼

        

    elif idx < 2 or idx > 2:
        labelimg.pack()
        for entry in entry_widgets:
            entry.place_forget()
        for label in entry_labels:
            label.place_forget()

        show_values_btn.pack_forget()
        predictlabel.place_forget()
        target_univ_label.pack_forget()
        yesButton.pack_forget()
        target_univ_combobox.pack_forget()

    if idx == 3:
        clear_entries()
        
        avg = avgGrade()
        avgtext.config(text = avg, font = "Arial", size = 20, weight = "bold")
        avglabel.place(x=100, y = 50)
        showlabel.place(x=100, y = 100)
        avgtext.place(x=250, y = 75)
    
        targetlabel.place(x=250, y = 130)
        tragetText.config(font = "Arial", size = 20, weight = "bold")
        tragetText.place(x=250, y = 125)
        showlabel2.place(x=250, y = 145)
        
        MatchingBtn.config(command = plusidx)
        MatchingBtn.place(x=250, y = 400)

    elif idx < 3 or idx > 3:
        avgtext.place_forget()
        avglabel.place_forget()
        showlabel.place_forget()
        showlabel2.place_forget()
        avgtext.place_forget()
        targetlabel.place_forget()
        tragetText.place_forget()
        MatchingBtn.place_forget()


    if idx == 4:
        update_buttons()

        pass
        
        
            
        
def show_values():
    """모든 엔트리 값 출력."""
    for i, var in enumerate(entry_vars):
        print(f"{entry_labels_file[i]}: {var.get()}")
        if var :
             print("True")
        else :
             print("False")
        print(type(entry_vars[i]))

def validate_entries(*args):
    """모든 엔트리 값이 입력되었는지 확인."""
    all_filled = all(var.get().strip() for var in entry_vars)
    if all_filled:
        yesButton.config(state=NORMAL, command=task)
   

def task() :
    global model, input_data
    update_buttons()
    
    predictlabel.config(text = "잠시만 기다려 주십시오 ...")

    target_univ = target_univ_var.get()
    target_univ_rank = university_rankings.get(target_univ, 0)

    try :
        input_data = [float(var.get()) if var.get().strip() != '' else 0.0 for var in entry_vars]
        input_data.append(float(target_univ_rank))  # 대학 랭크 추가
        input_data = np.array(input_data).reshape(1, -1)  # NumPy 배열로 변환
        res = model.predict(input_data)
    except ValueError:
        predictlabel.config(text="입력값에 숫자가 아닌 항목이 있습니다.")
        return
    try:
        input_data = scaler.transform(input_data)
        res = model.predict(input_data)
        predictlabel.config(
            text=f"{target_univ}의 합격률은 {res[0][0] * 100:.2f}% 입니다."
        )
    except Exception as e:
        predictlabel.config(text=f"예측 중 오류 발생: {str(e)}")

    
def avgGrade():
   global entry_vars
   sumvar = 0
   for var in entry_vars:
        try:
            value = float(var.get())  # StringVar에서 값을 가져와 변환
            sumvar += value
        except ValueError:
            print("숫자가 아닌 값이 입력되었습니다.")  # 예외 처리
            return None
        avg = sumvar / len(entry_vars)
        return avg
        
        
    




if __name__ == "__main__":
#데이터 로딩
# 모델 정의 및 학습


    Mainimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\002.png")
    Infoimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\003.png")
    entryimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\grade.png")
    matchingimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\matching.png")

    Mainimg = Mainimg.subsample(x=int(Mainimg.width() /width), y =int(Mainimg.height()/height))
    Infoimg = Infoimg.subsample(x=int(Infoimg.width() /width), y =int(Infoimg.height()/height))
    entryimg = entryimg.subsample(x=int(entryimg.width() /width), y =int(entryimg.height()/height))
    matchingimg = matchingimg.subsample(x=int(matchingimg.width() / 300), y =int(matchingimg.height()/ 200))


    imges =  [Mainimg,
            Infoimg,
            "",
            ""]

    #제일 처음 화면
    labelimg= Label(root, image = imges[idx])
    labelimg.pack()


    nextbtn = Button(root, text = '다음',  command = nextpage, state=NORMAL, width=5, height=1)
    nextbtn.place(x=250, y =550)
    prebtn = Button(root, text = '이전',  command = prepage, state=NORMAL, width=5, height=1)
    prebtn.place(x=200, y =550)

    #동의 버튼 2페이지
    agreeimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\동의함.PNG")
    width = 25
    height = 25
    agreeimge = agreeimg.subsample(x=int(agreeimg.width() / 1), y = int(agreeimg.height() / 1))
    agreebtn = Button(root, text="동의", command=lambda: CheckVar1.set(1), image = agreeimg)
    agreebtn.place_forget()

    #3페이지 idx == 2 
    grade1 = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\grade1.png"), 
    grade2 = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\grade2.png"), 
    grade3 = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\grade3.png"), 
    grade4 = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\grade4.png"), 
    grade5 = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\grade5.png"), 
    gradet = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\tutorlabel.png")

    entry_labels_file = [grade1, grade2, grade3, grade4, grade5, gradet]
    for file in entry_labels_file :
        file.subsample(x=int(file.width() /width), y =int(file.height()/height))
   
    

    for i, img in enumerate(entry_labels_file):
        var = StringVar()
        label = Label(root, image = img)
        entry = Entry(root, textvariable=var,)
        entry_labels.append(label)
        entry_widgets.append(entry)
        entry_vars.append(var) 

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'FNC'))
    sys.path.append(r"C:\Users\wawa2\OneDrive\바탕 화면\AI_serv_UI")
    sys.path.append(r"C:\Users\wawa2\OneDrive\바탕 화면\AI_serv_UI\FNC")
    import FNC.get_info as GIF

    _, unive_list, _ = GIF.get_data()

    def update_predictlabel(*args):
        selected_univ = target_univ_var.get()
        if selected_univ:
            predictlabel.config(text=f"목표대학 {selected_univ}의 합격률을 예측하시겠습니까?")
        else:
            predictlabel.config(text="목표 대학을 선택하십시오.")

    # 목표대학은 target_univ_var에 저장
    target_univ_var = StringVar()
    target_univ_label = Label(root, text="목표 대학을 선택하세요.")

    target_univ_var.trace_add("write", update_predictlabel)

    target_univ_combobox = ttk.Combobox(root, textvariable=target_univ_var, values=unive_list)
    show_values_btn = Button(root, text="입력값 출력", command=show_values)


    predictlabel = Label(root, text = f"목표대학 {target_univ_var}의 합격률을 예측하시겠습니까?")
    yesButton = Button (root, text = "예", state=NORMAL, width=5, height=1)

    #4페이지 idx == 3
    selected_univ = target_univ_var.get()
    avgimg = PhotoImage(file=r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\avgGrade.png")
    showimg = PhotoImage(file =r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\입력창.PNG" )
    targetimg = PhotoImage(file =r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\taget_unive.png" )

    avgimg.subsample(x=int(avgimg.width() /200), y =int(avgimg.height()/100))
    showimg.subsample(x=int(showimg.width() /200), y =int(showimg.height()/100))
    targetimg.subsample(x=int(targetimg.width() /750), y =int(targetimg.height()/500))

    avglabel = Label(root, image = avgimg)
    showlabel = Label(root, image = showimg)
    showlabel2 = Label(root, image = showimg)
    avgtext = Label(root,  text = '')

    targetlabel = Label(root, image = targetimg)
    tragetText = Label(root, text = selected_univ)
    MatchingBtn = Button(root, image = matchingimg)
    
    

    root.mainloop()



    
    
    
    



    

    
    



   
    
    
    
    



    
    

    
    
    
    
    
    
   
   

            
            
   