import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/eccv16/la_muse.t7')   ## 이미지를 로드해서 net에 저장, 배경이 될 사진

img = cv2.imread('imgs/test.jpg') # 01.jpg 불러오기, 변환할 사진

h, w, c = img.shape  # 전처리 코드
 
img = cv2.resize(img, dsize=(500, int(h / w * 500))) # 전처리 코드 / 500 * 500 으로 크기 조정

MEAN_VALUE = [103.939, 116.779, 123.680] # 전처리 코드  / 이미지 각 픽셀에서 해당 값을 빼서 성능을 높임
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE) # 전처리 코드  / 차원변형해서 blob에 담아줌

net.setInput(blob) # 전처리한 결과를 input에 저장
output = net.forward() # output에 추론한 결과를 저장

output = output.squeeze().transpose((1, 2, 0)) # 후처리 코드 / 차원 변형했던 것을 원래도로 되돌려 놓음.
output += MEAN_VALUE # 후처리 코드 /  전처리 할때 뺐던 MEAN_VALUE를 다시 넣어줌.

output = np.clip(output, 0, 255) # 후처리 코드 / 범위 넘어가는 값 잘라내기
output = output.astype('uint8') # 후처리 코드 / 이미지의 자료형은 정수형으로 바꿔줌(사람이 볼 수 있는 이미지 형태)

cv2.imshow('output', output) # output 이미지 출력
cv2.imshow('img', img)  # 이미지 띄우기
cv2.waitKey(0)