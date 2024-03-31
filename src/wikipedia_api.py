import wikipedia

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
      try:
        latitude =  wikipedia.page(get[0]).coordinates[0]
        longitude = wikipedia.page(get[0]).coordinates[1]
      except KeyError: 'coordinates'
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
