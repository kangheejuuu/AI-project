# sns(인스타그램)에서 크롤링을 통해 이미지 데이터를 얻는 코드입니다.

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

# 인스타그램 관련 코드는 여기부터
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(options=chrome_options)

# 인스타그램으로 이동
driver.get("https://instagram.com")

# 인스타그램 자동로그인
driver.implicitly_wait(10)
login_id = driver.find_element(By.CSS_SELECTOR, 'input[name="username"]')
login_id.send_keys('nae0_0429')
login_pwd = driver.find_element(By.CSS_SELECTOR, 'input[name="password"]')
login_pwd.send_keys('sodud2soddl!@')
driver.implicitly_wait(10)
login_id.send_keys(Keys.ENTER)

baseurl = 'https://www.instagram.com/explore/tags/'
plusurl = input('검색할 태그를 입력하세요: ')
url = baseurl + plusurl
print(url)
driver.get(url)
driver.implicitly_wait(15)

label = plusurl + "_label"
print(label)

# label을 int로 변환
label_int = eval(label)

# 첫번째 사진 누름
first_img = driver.find_element(By.CSS_SELECTOR, '._aagw').click()

driver.implicitly_wait(15)


for i in range(50):
    try:
        # 이미지를 모델에 전달하여 예측
        img_element = driver.find_element(By.CSS_SELECTOR, '._aatk .x5yr21d.xu96u03.x10l6tqk.x13vifvy.x87ps6o.xh8yej3')
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
        output_folder = f'predicted_images_{label}_insta'
        os.makedirs(output_folder, exist_ok=True)

       # 정확도가 1.5 이상이고, 예측된 클래스가 지정한 클래스 레이블과 동일한 경우에만 추가 작업 수행
        if predicted_class == label_int:
            print('Processing image', i)
            
            print('Download image from class', predicted_class)  # 검색하는 라벨에 맞게 변경

            # 이미지 저장
            img_filename = f'{i}_predicted_{predicted_class}.jpg'
            img_path = os.path.join(output_folder, img_filename)
            urllib.request.urlretrieve(img_src, img_path)
            print(f'Image saved to: {img_path}')


        # 다음 버튼 클릭
        driver.find_element(By.CSS_SELECTOR, '._aaqg ._abl-').click()

    except FileNotFoundError:
        # 이미지 파일이 없을 경우 처리
        print(f'Image {i} not found. Skipping...')
        # 다음 이미지 처리
        driver.find_element(By.CSS_SELECTOR, '._aaqg ._abl-').click()

    except NoSuchElementException:
        # 요소를 찾지 못할 경우 처리
        print(f'Element not found. Skipping...')
        # 다음 이미지 처리
        driver.find_element(By.CSS_SELECTOR, '._aaqg ._abl-').click()

    except Exception as e:
        # 기타 예외 처리
        print(f'Error processing image {i}: {str(e)}')
        # 다음 이미지 처리
        driver.find_element(By.CSS_SELECTOR, '._aaqg ._abl-').click()

    except:
        driver.find_element(By.CSS_SELECTOR, '._aaqg ._abl-').click()

driver.close()
