from google_images_download import google_images_download   #importing the library
from selenium import webdriver
import time
import os
import urllib
import requests
import uuid

dir_path = '~/Downloads/'

# Downloads image with a given url into the images folder
def download_image(url):
    print("[INFO] downloading {}".format(url))
    name = str(url.split('/')[-1])
    urllib.request.urlretrieve(url, 'images/' + str(uuid.uuid1()) + '.jpg')

driver = webdriver.Chrome()
driver.get('https://www.google.com/search?q=dog&rlz=1C5CHFA_enUS904US904&sxsrf=ALeKk01QYyYCWK4mCaqR32KvrYuonWvKpw:1591305046827&source=lnms&tbm=isch&sa=X&ved=2ahUKEwivhK6gienpAhUDna0KHczPACwQ_AUoAXoECBgQAw&biw=1280&bih=623')

#scroll down
driver.execute_script("window.scrollBy(0, 1000000)")

link_tags = driver.find_elements_by_tag_name('img')
print(len(link_tags))

#urls for all images
hrefs = []

for tag in link_tags:
    if tag.get_attribute("src") == None:
        print('found null')
    else:
        hrefs.append(tag.get_attribute("src"))

for href in hrefs:
    try:
        download_image(str(href))
        time.sleep(1)
    except Exception as e:
        print(e)
        pass

driver.close()