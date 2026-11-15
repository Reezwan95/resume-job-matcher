"""Minimal Flask front end: paste a resume and a job description, get a match report."""
from flask import Flask, render_template, request
from src.matcher import match

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        result = match(request.form.get("resume", ""), request.form.get("job", ""))
    return render_template("index.html", result=result, form=request.form)


if __name__ == "__main__":
    app.run(debug=True)
