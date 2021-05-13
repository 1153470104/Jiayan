# coding: utf-8
"""
some pre-processing script to process the tang poetry data
"""

# -------------- 处理得到曲江所有诗歌文本 --------------
# f = open("resource/small.txt", encoding='utf-8')
# newf = open("resource/small_raw.txt", 'w', encoding='utf-8')
# line = f.readline()
# while(line):
#     newf.write(line.replace(" ", ""))
#     line = f.readline()
# f.close()
# newf.close()

# -------------- 处理得到曲江所有诗歌文本 --------------
f = open("resource/guanzhong.txt", encoding='utf-8')
w1, w2, w3, w4 = 0, 0, 0, 0
line = f.readline()
while line:
    length = len(line)
    print(length)
    if length == 2:
        w1 += 1
    elif length == 3:
        w2 += 1
    elif length == 4:
        w3 += 1
    elif length == 5:
        w4 += 1
    line = f.readline()

f.close()
print('bb',  w1 / (w1+w2), '; bc', w2 / (w1+w2))
print('cb', w2/(w2+w3), 'cd', w3/(w2+w3))
print('db', w3/(w3+w4), 'de', w4/(w3+w4))
print('eb', 1, 'ee', 0)
