## Implementing Canny edge detector


### 개발 환경

- OS : Windows 11   
- IDE : Visual Studio Code
- Python-3.9.5


### 실행하는 법

1. 상단 Reading video 부분에 원하는 동영상의 경로 및 동영상 이름을 넣어줍니다.   
2. Sobel filter : Result 부분에서 sobel_out, Saving each processed pixels 부분에서 Sobel mask, 맨 아래에서 sobel_out 부분의 주석을 제거합니다.   
3. Non Maximum Suppression : Result 부분에서 nms_out, Saving each processed pixels 부분에서 Non Maximum Suppression, 맨 아래에서 nms_out 부분의 주석을 제거합니다.   
4. Hysteresis Thresholding : Result 부분에서 hys_out, Saving each processed pixels 부분에서 Hysteresis Thresholding, 맨 아래에서 hys_out 부분의 주석을 제거합니다. 두번째 parameter에는 high threshold, 마지막 parameter에는 low threshold를 넣어서 원하는 edge가 나오는 영상을 찾을 수 있습니다.