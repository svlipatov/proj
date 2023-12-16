from check_photo import *
from check_photo_model_init import *

model, cat = init_model()

Puskin_pamiatnik = Image.open("Data/Test_photo/1.jpg")
Nov_arbat1 = Image.open("Data/Test_photo/2.jpg")
Pushkin_dom1 = Image.open("Data/Test_photo/3.jpg")
CDA1 = Image.open("Data/Test_photo/4.jpg")
Okudjava1 = Image.open("Data/Test_photo/5.jpg")
Kinoteatr = Image.open("Data/Test_photo/6.jpg")
test_photos_dict = {'Puskin_pamiatnik':Puskin_pamiatnik,
                    'Nov_arbat1':Nov_arbat1,
                    'Pushkin_dom1': Pushkin_dom1,
                    'CDA1': CDA1,
                    'Okudjava1': Okudjava1,
                    'Kinoteatr': Kinoteatr,
                    }
for name in test_photos_dict:
    res_cat, res_score = check_photo1(model, cat, test_photos_dict[name])
    print(f"{res_cat}: {100 * res_score:.1f}%", "right answer", name)