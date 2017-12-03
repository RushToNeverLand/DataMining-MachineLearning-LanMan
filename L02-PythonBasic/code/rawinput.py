#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('GBK')  # 在当前笔记本电脑上
# sys.setdefaultencoding('utf-8')

age = int(raw_input(u"请输入你的年龄： "))

if age >= 18:
    print u"你是个adult."  
elif age >= 12:  # 可以有多个elif语句
    print "You are a teenager."
else:
    print u"你还是个小朋友呦."