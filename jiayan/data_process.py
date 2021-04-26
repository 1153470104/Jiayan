# coding: utf-8
"""
some pre-processing script to process the tang poetry data
"""

# -------------- 处理得到曲江所有诗歌文本 --------------
f = open("resource/small.txt", encoding='utf-8')
newf = open("resource/small_raw.txt", 'w', encoding='utf-8')
line = f.readline()
while(line):
    newf.write(line.replace(" ", ""))
    line = f.readline()
f.close()
newf.close()