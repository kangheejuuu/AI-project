from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import urllib.request
import ssl
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
from selenium.common.exceptions import NoSuchElementException
import os
import time

ssl._create_default_https_context = ssl._create_unverified_context

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

label = plusurl + "_label"
print(label)

# label을 int로 변환
label_int = eval(label)
    
    
for i in range(50):
    # 스크롤 다운하여 더 많은 트윗 로딩
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(10)  # 로딩 대기
        
    try:
        # 이미지를 모델에 전달하여 예측
        img_elements = driver.find_elements(By.CSS_SELECTOR, '.css-175oi2r.r-1kqtdi0.r-1phboty.r-rs99b7.r-1867qdf.r-1udh08x.r-o7ynqc.r-6416eg.r-1ny4l3l .css-9pa8cd')
        img_element = img_elements[i]  # 리스트에서 현재 인덱스에 해당하는 이미지 가져오기
        img_src = img_element.get_attribute('src')


        # 이미지를 모델에 전달하여 예측
        image = Image.open(urllib.request.urlopen(img_src))
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image_tensor = image_transform(image).unsqueeze(0)  # 배치 차원 추가
        with torch.no_grad():
            output = model(image_tensor)
        print('Model prediction successful.')

        # 예측 결과 출력
        print('Model output:', output)

        # 예측된 클래스 확인
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()
        print(f'Image {i}: Predicted Class - {predicted_class}')

        # 예측된 클래스별로 이미지를 저장할 폴더 생성
        output_folder = f'predicted_images_{plusurl}_twitter'
        os.makedirs(output_folder, exist_ok=True)

        # 정확도가 1.5 이상이고, 예측된 클래스가 지정한 클래스 레이블과 동일한 경우에만 추가 작업 수행
        if output.max().item() >= 1.5 and predicted_class == label_int:
            print('Processing image', i)
            print('Download image from class', predicted_class)  # 검색하는 라벨에 맞게 변경

            # 이미지 저장
            img_filename = f'{i}_predicted_{predicted_class}.jpg'
            img_path = os.path.join(output_folder, img_filename)
            urllib.request.urlretrieve(img_src, img_path)
            print(f'Image saved to: {img_path}')

    except FileNotFoundError:
        # 이미지 파일이 없을 경우 처리
        print(f'Image {i} not found. Skipping...')
    
    except NoSuchElementException:
        # 요소를 찾지 못할 경우 처리
        print(f'Element not found. Skipping...')
    
    except Exception as e:
        # 기타 예외 처리
        print(f'Error processing image {i}: {str(e)}')

driver.close()
