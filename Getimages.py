from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.mkdir(directory)
    except OSError:
        print("Error: Failed to create the directory")

def crawling_img(name):
    driver = webdriver.Chrome()
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
    elem = driver.find_element(By.NAME, "q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)
    
    SCROLL_PAUSE_TIME = 1
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
            except:
                break
        last_height = new_height

    imgs = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")
    print(imgs)
    save_folder = os.path.join("C:/disasters", name)
    createDirectory(save_folder)
    
    count = 0
    for img in imgs:
        try:
            count += 1
            driver.execute_script("arguments[0].click();", img)
            time.sleep(2)
            
            # 이미지 URL 가져오기
            img_url = driver.find_element(By.CSS_SELECTOR, ".n3VNCb").get_attribute("src")
            
            timestamp = int(time.time())  # 현재 시간을 타임스탬프로 사용하여 중복을 피함
            file_name = f"{name}_{count}_{timestamp}.jpg"
            
            path = os.path.join(save_folder, file_name)
            urllib.request.urlretrieve(img_url, path)
            print(f"Downloaded: {path}")
            
            if count >= 5:
                break
        except Exception as e:
            print(f"Error: {e}")
            pass

    driver.close()

disasters = ["typhoon"]

for disaster in disasters:
    crawling_img(disaster)
