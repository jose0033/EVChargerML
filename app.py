from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        Holiday = request.form.get("Holiday")
        Weather = request.form.get("Weather")
        Volume = request.form.get("Volume")
        Speed = request.form.get("Speed")
        Accident = request.form.get("Accident")

        print(Holiday,Weather,Volume,Speed,Accident)
        model1 = joblib.load("CART")
        #FLOAT
        Holiday = float(Holiday)
        Weather = float(Weather)
        Volume = float(Volume)
        Speed = float(Speed)
        Accident = float(Accident)


        pred1 = model1.predict([[Holiday,Weather,Volume,Speed,Accident]])
        model2= joblib.load("RF")
        pred2 = model2.predict([[Holiday,Weather,Volume,Speed,Accident]])
        model3= joblib.load("GB")
        pred3 = model3.predict([[Holiday,Weather,Volume,Speed,Accident]])
        return(render_template("index.html", result1=pred1, result2=pred2, result3=pred3))
    else:
        return(render_template("index.html", result1="Please Enter Details", result2="Please Enter Details", result3='Please Enter Details'))

if __name__ == "__main__":
    app.run()
