from backend import app

if __name__ == '__main__':
    with app.app_context():
        app.run(host='127.0.0.1',port=8765,debug=True)