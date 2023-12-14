from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import urllib.request
from PIL import Image
import time
import os


# 웹 드라이버 설정
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(options=chrome_options)

# 트위터로 이동
driver.get("https://twitter.com/i/flow/login")

# # 로그인 관련 코드
driver.implicitly_wait(10)
login_id = driver.find_element(By.CSS_SELECTOR, 'input[name="text"]')
login_id.send_keys('dellrow_Loh')
login_id.send_keys(Keys.ENTER)
login_pwd = driver.find_element(By.CSS_SELECTOR, 'input[name="password"]')
login_pwd.send_keys('qwer1234!')
driver.implicitly_wait(10)
login_pwd.send_keys(Keys.ENTER)

# 검색어 입력 (최신순)
baseurl = 'https://twitter.com/search?q='
plusurl = input('검색할 태그를 입력하세요: ')
baseurll = '&src=typed_query&f=live'
url = baseurl + plusurl + baseurll
print(url)
driver.get(url)
driver.implicitly_wait(15)

# # 검색어 입력 (인기순)
# baseurl = 'https://twitter.com/search?q='
# plusurl = input('검색할 태그를 입력하세요: ')
# baseurll = '&src=typed_query&f=top'
# url = baseurl + plusurl + baseurll
# print(url)
# driver.get(url)
# driver.implicitly_wait(15)



# 이미지 다운로드를 위한 디렉토리 생성
output_directory = f'{plusurl}_images'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 스크롤 다운하여 더 많은 트윗 로딩
for _ in range(5):  # 예시로 5번 스크롤 다운
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # 로딩 대기

# 이미지 다운로드
image_elements = driver.find_elements(By.CSS_SELECTOR, '.css-175oi2r.r-1kqtdi0.r-1phboty.r-rs99b7.r-1867qdf.r-1udh08x.r-o7ynqc.r-6416eg.r-1ny4l3l .css-9pa8cd')
for i, img_element in enumerate(image_elements):
    img_src = img_element.get_attribute('src')
    try:
            # 이미지 다운로드
            urllib.request.urlretrieve(img_src, os.path.join(output_directory, f'tweet_{i}.jpg'))
            print(f'Downloaded image {i}')
    except Exception as e:
            print(f'Error downloading image {i}: {str(e)}')

# 웹 드라이버 종료
driver.quit()
