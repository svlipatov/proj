import requests
from bs4 import BeautifulSoup
import os

def save_images(save_dir, keywords):
  os.makedirs(save_dir, exist_ok=True)
  for keyword in keywords:
      url = f"https://www.google.com/search?q={keyword}&tbm=isch"
      res = requests.get(url)
      soup = BeautifulSoup(res.text, "html.parser")
      img_tags = soup.find_all("img")
      for i, img in enumerate(img_tags):
          try:
              img_url = img["src"]
              res = requests.get(img_url)
              with open(f"{save_dir}/{keyword}{str(i).zfill(5)}.jpg", "wb") as f:
                  f.write(res.content)
          except:
              continue

keywords = ["cat"]
save_dir = "train"
save_images(save_dir, keywords)