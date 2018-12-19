#!/usr/bin/env python
# encoding=utf-8
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import time

'''中文图片生产器'''


class getName(object):
    # 名称,索引生成
    def __init__(self):
        self.fontindex = []  # 字典索引
        self.num = 0  # 字典字数
        # 字体种类
        self.font = ['msyh.ttf', 'msyhbd.ttf', 'simhei.ttf', 'SIMLI.ttf', 'STLITI.ttf', 'simsun.ttf', 'STXINGKA.ttf']
        self.fontsize = 20  #字体大小
        self.sonfilename=""#图片子文件夹名字

    def getFont(self):
        # 获取字体
        # temp = np.random.randint(0, 6)
        return self.font[5]

    def getfontsize(self):
        # 获取字体大小
        # self.fontsize = np.random.randint(15, 30)
        self.fontsize = 20
        return self.fontsize

    def getNum(self):
        # 获取字数
        return self.num

    def setNum(self):
        # 设置字数
        self.num = np.random.randint(1, 6)
        # self.num=8

    def getIndex(self):
        self.fontindex = []
        # 获取字典索引
        for i in range(self.num):
            temp = np.random.randint(0, 100)#字符数量
            self.fontindex.append(temp)
        return self.fontindex

    def getsignIndex(self):
        # 获取标点索引
        signindex = []
        for i in range(2):
            temp = np.random.randint(0, 165)
            signindex.append(temp)
        return signindex

    def setsonfilename(self,name):
        #设置图片子文件夹名字
        self.sonfilename=name

    def getsonfilename(self):
        #获取图片子文件夹名字
        return self.sonfilename


class imageBuilder(object):
    # 图片生成器
    def __init__(self, width, height, filename):
        self.w = width
        self.h = height
        self.filename = filename
        self.tt = GN.getFont()
        self.fontsize = GN.getfontsize()
        self.num = GN.getNum()

    def getposition(self):
        size = self.fontsize
        width = self.w
        height = self.h
        num=self.num
        endy = int((height - size) / 2)
        endx = int((width - size*num ) / 2)
        return endx, endy

    def buider(self):
        # 背景生成
        p = Image.new('RGBA', (self.w, self.h), (255, 255, 255))
        return p

    # 获得颜色
    def getRandomColor(self):
        return (np.random.randint(30, 100), np.random.randint(30, 100), np.random.randint(30, 100))

    def fontpaste(self, text):
        # 文本粘贴
        ttfont = ImageFont.truetype("./fonts/" + self.tt, self.fontsize)
        x, y = self.getposition()
        im = self.buider()
        draw = ImageDraw.Draw(im)
        # print x
        draw.text((x, y), unicode(text,"utf-8"), fill=(0, 0, 0), font=ttfont)

        if not os.path.exists("./images"):
            os.mkdir('images')

        imgdir=GN.getsonfilename()

        if not os.path.exists("./images/"+str(imgdir)):
            os.mkdir('./images/'+imgdir)

        filepath = './images/'+imgdir+"/"+ self.filename
        im=im.convert("RGB")
        im.save(filepath, 'JPEG')

        with open("./images/"+imgdir+".txt", "a") as datatxt:
            txt = self.filename + " " + text+"\n"
            datatxt.write(txt)


def textBuider():
    # 获取文本
    GN.setNum()
    index = GN.getIndex()



    with open("charsets/charsets.txt") as file:
        lines = file.readlines()

    strtext = ''
    for i in index:
        strtext += lines[i].split(" ")[1].split("\r\n")[0]

    return strtext


if __name__ == "__main__":
    stattime = time.time()
    global GN
    GN = getName()

    count=0
    num=6400 # 所生成图片的个数
    batch=1 #文件批数
    imgnameindex=1
    for index in range(0, num):
        imgname = "%06d" % index + '.jpg'  # 图片的名称
        imgdir = "img%03d"%imgnameindex  # 图片所在子文件夹的名称

        GN.setsonfilename(imgdir)
        string = textBuider()
        imgs = imageBuilder(100, 32, imgname)
        imgs.fontpaste(string)

        count += 1
        if count>=num//batch:
            count=0
            imgnameindex+=1

    endtime = time.time()
    print("success!" + " Spend time:" + str(endtime - stattime) + "秒")
