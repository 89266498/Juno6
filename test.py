# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
proxies = {'http': "127.0.0.1:8118",
           'https': "127.0.0.1:8118"} #"socks5://127.0.0.1:1080"
r = requests.get('https://www.google.com/',proxies=proxies)
print(r.status_code)
print(BeautifulSoup(r.content, 'lxml').title.text)