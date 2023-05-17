from flask import Flask, render_template,request,jsonify

import mysql.connector
from mysql.connector import Error

DEBUG = False

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add')
def redira():
    return render_template('add.html')
@app.route('/six')
def redir6():
    return render_template('six.html')

@app.route('/five')
def redir5():
    return render_template('five.html')

@app.route('/crud')
def redirc():
    return render_template('crud.html')

@app.route('/SP')
def redircSP():
    return render_template('sp.html')
@app.route('/query',methods=['POST'])
def do_query():
    if not DEBUG:
        try:
            cursor.execute(request.json["query"])
            result = cursor.fetchall()
            connection.commit()
        except Error as e:
            print(f"error '{e}'")
            return jsonify({"result":False,"msg":f"error: '{e}'"})
    else:
        print("Did query: ",request.json["query"])
        return jsonify({"result":False,"msg":["1","2","3",["1","2"]]})
    return jsonify({"result":True,"msg":result})
        
if __name__ == '__main__':
    if not DEBUG:
        print("Trying to connect to database...")
        try:
            connection = mysql.connector.connect(host="34.71.105.198", user="root", passwd="12345678",database='testdatabase')
        except Error as e:
            print(f"error '{e}'")
        
        print("connected")
        cursor = connection.cursor()
    app.run(host='0.0.0.0',port=5000,debug = False)
    if not DEBUG:
        connection.close()
    
    