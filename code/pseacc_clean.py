import re
import pandas as pd

complete_list = []
with open('pseaac2.csv', 'r') as q:
    line = 0
    while line < 17500:
        sub_list = []
        x = q.readline()
        if x.startswith("N") or x.startswith("\"N"):
            x = re.sub('[^A-Za-z0-9\s]', '', x)
            x = x[5:]
            x = x.strip()
            sub_list.append(x)
            q.readline()
            row1 = q.readline()
            row1 = re.sub('[^0-9.,]', '', row1)
            row1 = row1.split(',')
            row2 = q.readline()
            row2 = re.sub('[^0-9.,]', '', row2)
            row2 = row2.split(',')
            row1.extend(row2)
            row3 = q.readline()
            row3 = re.sub('[^0-9.,]', '', row3)
            row3 = row3.split(',')
            row1.extend(row3)
            row4 = q.readline()
            row4 = re.sub('[^0-9.,]', '', row4)
            row4 = row4.split(',')
            row1.extend(row4)
            row5 = q.readline()
            row5 = re.sub('[^0-9.,]', '', row5)
            row5 = row5.split(',')
            row1.extend(row5)
            sub_list.extend(row1)
            complete_list.append(sub_list)
            line += 1
        else:
            continue
        line += 1
q.close()

df = pd.DataFrame(complete_list,
 columns = ['Name', 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50'])

df['Known_Allergen'] = 0

i = 0 
while i <= 2530:
    df.at[i,'Known_Allergen']= 1
    i += 1


df.to_csv('clean_data.csv', index=False)