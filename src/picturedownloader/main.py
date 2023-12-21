from icrawler.builtin import BingImageCrawler
import os

imageFolder = 'images'


def download_images(imageFolder, query, limit):
    imageFolder=os.path.join(imageFolder, query)
    os.makedirs(name=imageFolder,
                exist_ok=True)
    google_crawler = BingImageCrawler(parser_threads=1,
                                      downloader_threads=1,
                                      storage={'root_dir': imageFolder})
    # Parameters can be found in the icrawler documentation
    # https://icrawler.readthedocs.io/en/latest/builtin.html
    filters = dict(
        type="photo",
        size='large',
        date="pastyear")
    google_crawler.crawl(keyword=query,
                         max_num=limit,
                         filters=filters)
    return os.listdir(imageFolder)


# Задаем список достопримечательностей и количество изображений, которые нужно загрузить
sights = [
    "Кинотеатр Художественный на Арбате",
    "Театр им. Вахтангова",
    "Центральный Дом Актера на Арбате",
    "Мемориальная квартира А.С. Пушкина на Арбате",
    "Памятник Пушкину и Гончаровой на Арбате",
    "Памятник Окуджаве на Арбате",
    "Хард-рок кафе на Арбате",
    "Дома-книжки на Новом Арбате"
]
num_images = 200

for sight in sights:
    print(f"Загрузка изображений достопримечательности '{sight}':")
    image_paths=download_images(imageFolder, sight, num_images)
    print(f"Загружено {len(image_paths)} изображений\n")
