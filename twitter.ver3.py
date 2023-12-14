from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager  #크롬 드라이버 자동 업데이트
import urllib.request
import ssl
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
from selenium.common.exceptions import NoSuchElementException
import os
from selenium.webdriver.common.action_chains import ActionChains

ssl._create_default_https_context = ssl._create_unverified_context
import time

# 모델 불러오기
class DisasterDetectionModel(nn.Module):
    def __init__(self):
        super(DisasterDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 112 * 112, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = DisasterDetectionModel()
model.load_state_dict(torch.load('disaster_detection_model2.pth'))
model.eval()

# 클래스 레이블 정의
earthquake_label = 0
fire_label = 1
flood_label = 2
landslide_label = 3
typhoon_label = 4

# 웹 드라이버 설정
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(options=chrome_options)

# 트위터로 이동
driver.get("https://twitter.com/i/flow/login")

# # 로그인 관련 코드 (로그인이 필요한 경우)
driver.implicitly_wait(10)
login_id = driver.find_element(By.CSS_SELECTOR, 'input[name="text"]')
login_id.send_keys('dellrow_Loh')
login_id.send_keys(Keys.ENTER)
login_pwd = driver.find_element(By.CSS_SELECTOR, 'input[name="password"]')
login_pwd.send_keys('qwer1234!')
driver.implicitly_wait(10)
login_pwd.send_keys(Keys.ENTER)

# 검색어 입력
baseurl = 'https://twitter.com/search?q='
plusurl = input('검색할 태그를 입력하세요: ')
baseurll = '&src=typed_query&f=live'
url = baseurl + plusurl + baseurll
print(url)
driver.get(url)
driver.implicitly_wait(15)

# 이미지 다운로드를 위한 디렉토리 생성
output_directory = f'{plusurl}_images'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for i in range(50):
    
    # 스크롤 다운하여 더 많은 트윗 로딩
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)  # 로딩 대기
    
    try:
        # 동영상 클래스를 이용하여 동영상 요소 찾기
        video_element = driver.find_element(By.CSS_SELECTOR, '.css-175oi2r.r-1ssbvtb.r-1s2bzr4')

        # 동영상의 각 프레임 URL 가져오기
        frame_elements = video_element.find_elements(By.TAG_NAME, 'img')
        
        # 프레임별로 처리
        frame_count = 0
        for frame_element in frame_elements:
            # 프레임을 PIL 이미지로 변환
            frame_url = frame_element.get_attribute('src')
            pil_image = Image.open(urllib.request.urlopen(frame_url))

            # 이미지를 모델에 전달하여 예측
            image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image_tensor = image_transform(pil_image).unsqueeze(0)  # 배치 차원 추가
            with torch.no_grad():
                output = model(image_tensor)

            # 예측 결과 출력
            print('Model output:', output)

            # 예측된 클래스 확인
            _, predicted_class = torch.max(output, 1)
            predicted_class = predicted_class.item()
            print(f'Frame {frame_count}: Predicted Class - {predicted_class}')

            # 예측된 클래스에 따라 프레임 저장
            if output.max().item() >= 1.5:
                frame_filename = f'frame_{frame_count}_predicted_{predicted_class}.jpg'
                pil_image.save(frame_filename)
                print(f'Frame saved to: {frame_filename}')

            frame_count += 1


    except FileNotFoundError:
        # 미디어 파일이 없을 경우 처리
        print(f'Media {i} not found. Skipping...')

    except NoSuchElementException:
        # 요소를 찾지 못할 경우 처리
        print(f'Element not found. Skipping...')

    except Exception as e:
        # 기타 예외 처리
        print(f'Error processing media {i}: {str(e)}')

# 자원 해제
driver.quit()