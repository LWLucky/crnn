import sys
import glob
f=open("../datasetbuilder/images/img001.txt","r")
f1=open("./img.txt","w")
f2=open("./label.txt","w")
for i in f.readlines():
    img="../datasetbuilder/images/img001/"+i.split(" ",1)[0]+"\n"
    label=i.split(" ",1)[1]
    f1.write(img)
    f2.write(label)
f.close()
f1.close()
f2.close()
'''f=open("./charset.txt","r")
alphabet1=[]

for i in f.readlines():
    alphabet1.append(unicode(i.split(" ",1)[1].strip(),"utf8"))
alphabet="".join(alphabet1)
#alphabet.encode("utf-8")
nclass = len(alphabet) + 1
print len(alphabet)
print alphabet
print nclass
f.close()'''

'''alphabet1=[]
with open("./charset.txt",'r') as file:
    data=file.readlines()
    for i in range(len(data)):
        alphabet1.append(unicode(data[i].split(" ")[1].split("\r\n")[0],"utf-8"))
alphabet="".join(alphabet1)
print alphabet1'''
