from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        x = request.form.get("x")
        y = request.form.get("y")
        congestion = request.form.get("congestion")
        satisfaction = 1#request.form.get("satisfaction")
        accident = 1 #request.form.get("accident")

        print(x,y,congestion,satisfaction,accident)
        model1 = joblib.load("CART")
        x = float(x)
        y = float(y)
        congestion = float(congestion)
        satisfaction = float(satisfaction)
        accident = float(accident)


        pred1 = model1.predict([[x,y,congestion,satisfaction,accident]])
        model2= joblib.load("RF")
        pred2 = model2.predict([[x,y,congestion,satisfaction,accident]])
        model3= joblib.load("GB")
        pred3 = model3.predict([[x,y,congestion,satisfaction,accident]])
        return(render_template("index.html", result1=pred1, result2=pred2, result3=pred3))
    else:
        return(render_template("index.html", result1="Please Enter Details", result2="Please Enter Details", result3='Please Enter Details'))

if __name__ == "__main__":
    app.run()
