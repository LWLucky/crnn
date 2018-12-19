import glob
f=open("./charsets/charsets.txt")
a=[]
for i in f.readlines():
    a.append(i.split(" ",1)[1].strip())
b="".join(a)
b.decode("utf-8")
print b