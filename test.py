# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from browsermobproxy import Server
from selenium import webdriver


proxies = {'http': "127.0.0.1:8118",
           'https': "127.0.0.1:8118"} #"socks5://127.0.0.1:1080"
r = requests.get('https://www.pornhub.com/view_video.php?viewkey=ph5f85fcebef03c',proxies=proxies)
print(r.status_code)

# soup = BeautifulSoup(r.content, 'lxml')
# print(soup.title.text)


# Purpose of this script: List all resources (URLs) that
# Chrome downloads when visiting some page.

### OPTIONS ###
url = 'https://www.pornhub.com/view_video.php?viewkey=ph5f85fcebef03c'
chromedriver_location = "./chromedriver" # Path containing the chromedriver
browsermobproxy_location = "/opt/browsermob-proxy-2.1.4/bin/browsermob-proxy" # location of the browsermob-proxy binary file (that starts a server)
chrome_location = "/usr/bin/x-www-browser"
###############

# Start browsermob proxy
server = Server(browsermobproxy_location)
server.start()
proxy = server.create_proxy()

# Setup Chrome webdriver - note: does not seem to work with headless On
options = webdriver.ChromeOptions()
options.binary_location = chrome_location
# Setup proxy to point to our browsermob so that it can track requests
options.add_argument('--proxy-server=%s' % proxy.proxy)
driver = webdriver.Chrome(chromedriver_location, chrome_options=options)

# Now load some page
proxy.new_har("Example")
driver.get(url)

# Print all URLs that were requested
entries = proxy.har['log']["entries"]
for entry in entries:
    if 'request' in entry.keys():
        print entry['request']['url']

server.stop()
driver.quit()