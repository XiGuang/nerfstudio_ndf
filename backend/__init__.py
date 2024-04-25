from flask import Flask
from backend.views import main

app = Flask(__name__)
app.register_blueprint(main)