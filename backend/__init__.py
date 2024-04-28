from flask import Flask
from backend.views import main

app = Flask(__name__, static_folder='static')
app.register_blueprint(main)