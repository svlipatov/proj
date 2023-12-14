from transformers import AutoTokenizer, T5ForConditionalGeneration

def init_model_and_tokenizer():
    """Initializes the model and the tokenizer."""
    model_name = "IlyaGusev/rut5_base_sum_gazeta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    return model, tokenizer

def summarize(text, model, tokenizer):
    """Summarizes the text."""
    input_ids = tokenizer(
        [text],
        max_length=600,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]
    
    output_ids = model.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )[0]
    
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    # Test
    text = "Памятник Александру Пушкину и Наталье Гончаровой', 'summary': 'Па́мятник Алекса́ндру Пу́шкину и Ната́лье Гончаро́вой — памятник известному русскому поэту Александру Пушкину и его жене Наталье Гончаровой. Установлен в 1999 году на Арбате напротив дома, где они жили. Авторами проекта являются скульпторы Александр и Игорь Бургановы, архитекторы Евгений Розанов и Е. К. Шумов.\nСкульптура копирует сцену после венчания пары и изображает их шагающими вперёд держась за руки. Обе бронзовые статуи выполнены с большим портретным сходством и установлены на гранитный постамент с надписью: «Александр Пушкин и Наталья Гончарова»."
    model, tokenizer = init_model_and_tokenizer()
    print(summarize(text, model, tokenizer))
