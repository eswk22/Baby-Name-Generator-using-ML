# import libraries
import requests
import lxml.html
from bs4 import BeautifulSoup
import csv

gender = ['boy','girl']
alpha = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
names = []
for g in gender:
    for alp in alpha:
        urlPage = "http://babynames.extraprepare.com/" + g + "-" + alp + ".php"
        # query the website and return the html to the variable ‘page’
        page = requests.get(urlPage)
        contents = page.content

        # parse the html using beautiful soup and store in variable `soup`
        soup = BeautifulSoup(contents, 'html.parser')
        names.extend(soup.find_all('h3'))
    with open('data.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        for name in names:
            writer.writerow([name,g])

