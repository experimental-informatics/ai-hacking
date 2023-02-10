import glob
with open('mergetexts_230203-1.csv', 'a') as csv_file:
    for path in glob.glob('*.txt'):
        with open(path) as txt_file:
            txt = txt_file.read() + '\n'
            csv_file.write(txt)

""" import os
import glob
with open('mergetexts_230203.csv', 'a') as csv_file:
    for path in glob.glob('/home/student/Dokumente/ai-hacking/hacking/lisa/gen_text/*.txt'):
        with open(path) as txt_file:
            txt = txt_file.read() + '\n' + os.path.splitext("/home/student/Dokumente/ai-hacking/hacking/lisa/gen_text/*.txt")[0] 
            csv_file.write(txt) """


""" import os
print(os.path.splitext("/home/student/Dokumente/ai-hacking/hacking/lisa/gen_text/*.txt")[0]) """