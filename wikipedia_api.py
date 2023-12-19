import wikipedia
from bs4 import BeautifulSoup as bs

wikipedia.set_lang("ru")

#Принимает список поиска
#Возвращает список словарей {'find':строка поиска,'summary': краткое содержание,
#                            'latitude': широта, 'longitude': долгота, '':}
def getWikipedia(pList):

  result = []
  for i in pList:

    latitude, longitude = '', ''

    get = wikipedia.search(i)
    if not get is None:

      summary = wikipedia.summary(get[0])

      page  = wikipedia.page(get[0])
      soup = bs(page.html(), 'html.parser')
      find = soup.find("a", {"class": "mw-kartographer-maplink"})
      if not find is None:
         latitude =  find['data-lat']
         longitude = find['data-lon']
    else:
      summary = 'Ничего не найдено'
      
    res = {'find':i,
             'summary': summary,
             'latitude': latitude,
             'longitude': longitude,
             }
    result.append(res)

  return result

if __name__ == "__main__":

  print(getWikipedia(["Новый Арбат"]))
