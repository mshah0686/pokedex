# Google images scraping script
# Author: Malav Shah
# Wrote to scrape images of the web with a query, but realized some road blocks with encryptions.
# Found API on google that was much easier, but will fix this up later to solve encryption issues

from google_images_download import google_images_download   #importing the library
from selenium import webdriver
import time
import os
import urllib
import requests
import uuid
import certifi
from selenium.webdriver.common.keys import Keys

dir_path = '~/Downloads/'

# Downloads image with a given url into the images folder
def download_image(url, category):
    print("[INFO] downloading {}".format(url))
    name = str(url.split('/')[-1])
    urllib.request.urlretrieve(url,store_folder + '/' + category + '/' + str(uuid.uuid1()) + '.jpg')


def get_urls(keywords):
    driver = webdriver.Chrome()
    for keyword in keywords:
        driver.get('https://www.google.com/')
        search = driver.find_element_by_name('q')
        search.send_keys(keyword, Keys.ENTER)

        elem = driver.find_element_by_link_text('Images')
        elem.get_attribute('href')
        elem.click()

        #scroll down
        #need to implement clicking load more results if more scraping needed
        for i in range(20):
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
        
        #make storage folder
        try:
            os.mkdir(store_path + '/' + keyword)
        except:
            pass

        for href in hrefs:
            try:
                download_image(str(href), keyword)
            except Exception as e:
                print(e)
                pass

    driver.close()

store_folder = 'scarped_images'
store_path = os.getcwd() + '/' + store_folder

if __name__ == '__main__':
    cwd = os.getcwd()
    try:
        os.mkdir(cwd + '/' + store_folder)
    except:
        pass

    search_words = ['balbasuar', 'charmander', 'pikachu', 'squirtle']
    get_urls(search_words)