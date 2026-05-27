from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

app = FastAPI(
    title="Titanic Survival API",
    description="API predicting Titanic passenger survival",
    version="1.0"
)

# Wczytanie modelu
filename = "model.h5"
model = pickle.load(open(filename, 'rb'))


class Passenger(BaseModel):
    pclass: int
    age: int
    sibsp: int
    parch: int
    fare: float
    sex: int
    embarked_q: int
    embarked_s: int


@app.get("/")
def home():
    return {"message": "Titanic API works"}


@app.post("/predict")
def predict(passenger: Passenger):

    data = [[
        passenger.pclass,
        passenger.age,
        passenger.sibsp,
        passenger.parch,
        passenger.fare,
        passenger.sex,
        passenger.embarked_q,
        passenger.embarked_s
    ]]

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    result = "Przeżyje" if prediction[0] == 1 else "Nie przeżyje"

    return {
        "prediction": int(prediction[0]),
        "result": result,
        "confidence": float(max(probability[0]) * 100)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)