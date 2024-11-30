from tkinter import *
from PIL import ImageTk, Image

root = Tk()


windows = root.geometry("600x700+150+150")

height = 450
width = 450

CheckVar1 = IntVar()

page = 1

def nextpage() :
    pass







Mainimg = PhotoImage(file = r"Images\002.png")
Infoimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\002.png")
entryimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\002.png")
predictimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\002.png")
infoentryimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\002.png")
matchingimg = PhotoImage(file = r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\002.png")

Mainimg = Mainimg.subsample(x=width, y =height)
Infoimg = Infoimg.subsample(x=width, y =height)
entryimg = entryimg.subsample(x=width, y =height)
predictimg = predictimg.subsample(x=width, y =height)
infoentryimg = infoentryimg.subsample(x=width, y =height)
matchingimg = matchingimg.subsample(x=width, y =height)

mainlabel = Label(root, image = Mainimg)
mainlabel.place(x=65, y = 150)



Label(text = "과외쌤 찾기", font = ('consolas', 20)).pack(fill='y')
Label(text = "확률 예측", font = ('consolas', 20)).pack(fill='y')
Label(text = "성적 기입란", font = ('consolas', 20)).pack(fill='y')
Label(text = "개인정보 동의란", font = ('consolas', 20)).pack(fill='y')
Label(text = "메인화면", font = ('consolas', 20)).pack(fill='y')

root.mainloop()

exit()

nextbtn = Button(root, text = '다음',  command = nextpage, state=NORMAL).pack(side = 'right')
#prebtn = Button(root, text = '이전',  command = prepage, state=NORMAL).pack(side = 'right')


        

agreeimg = Image.open(r"C:\Users\wawa2\OneDrive\Desktop\AI project\AIpredict_sys\AI_serv_UI\Images\동의함.PNG")
agreeimg = agreeimg.resize((50, 50))
agreeimg = ImageTk.PhotoImage(agreeimg)


agreelabel = Label(root, image = agreeimg)
agreelabel.image = agreeimg
agreelabel.place(x = 67.5, y = 100)

        




agreebtn = Checkbutton(
root, 
image = agreeimg,
variable=CheckVar1)
agreebtn.image = agreeimg
agreebtn.place(x=150, y=550)
        





 

def Pageup():
    global page
    
    if CheckVar1.get()==0 and page==2 :
        page = 2 
    
   
   
        
disagreelabel = Label(root, text = "동의해야 서비스 이용 가능합니다.")
disagreelabel.place(x=300, y=575)

    
                

                    
    
         
    



