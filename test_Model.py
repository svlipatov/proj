from src.main import load_recognition_model

model = load_recognition_model()

def test_load_recognition_model():
    assert model is not None
