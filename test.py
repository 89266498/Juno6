# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from browsermobproxy import Server
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

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
driver_location = "/web/juno/juno/Juno6/geckodriver-v0.24.0-linux64/geckodriver" # Path containing the chromedriver
browsermobproxy_location = "./browsermob-proxy-2.1.4-bin/browsermob-proxy-2.1.4/bin/browsermob-proxy" # location of the browsermob-proxy binary file (that starts a server)
firefox_binary = "/usr/bin/firefox"
###############

# Start browsermob proxy
# server = Server(browsermobproxy_location)
# server.start()
# proxy = server.create_proxy()

# Setup Chrome webdriver - note: does not seem to work with headless On
options = webdriver.FirefoxOptions()
options.binary = firefox_binary
# Setup proxy to point to our browsermob so that it can track requests
#options.add_argument('--proxy-server=%s' % proxy.proxy)
cap = DesiredCapabilities().FIREFOX.copy()
cap["marionette"] = False
# fp = webdriver.FirefoxProfile()
# # Here "2" stands for "Automatic Proxy Configuration"

# fp.set_preference('network.proxy.type',1)
# fp.set_preference('network.proxy.https', '127.0.0.1') 
# fp.set_preference('network.proxy.port', '8118') 
driver = webdriver.Firefox( capabilities=cap, options=options, executable_path="/web/juno/juno/Juno6/geckodriver-v0.24.0-linux64/geckodriver")
#firefox_profile=fp,
#driver.get('http://www.baidu.com')

#Now load some page
#proxy.new_har("Example")
driver.get('http://www.baidu.com')

# Print all URLs that were requested
# entries = proxy.har['log']["entries"]
# for entry in entries:
#     if 'request' in entry.keys():
#         print(entry['request']['url'])

#server.stop()
driver.quit()