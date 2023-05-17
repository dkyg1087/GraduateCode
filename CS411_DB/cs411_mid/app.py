from flask import Flask, render_template,request,jsonify

import mysql.connector
from mysql.connector import Error

DEBUG = False

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/advQ1',methods = ['GET'])
def advance_query1():
    if not DEBUG:
        cursor.execute("SELECT SUM(v.Views),t.Name FROM Describes d JOIN Videos v ON(d.VideoID=v.VideoID) JOIN Tags t ON (d.TagsID=t.TagsID) GROUP BY d.TagsID,t.Name ORDER BY SUM(v.Views) DESC LIMIT 15;")
        result = cursor.fetchall()
    else:
        result = [('8559102348', '[None]'), ('2629037028', 'funny'), ('1314675381', 'minecraft'), ('1304252895', 'challenge'), ('1216853513', 'BTS')]
    return result
@app.route('/advQ2',methods = ['GET'])
def advance_query2():
    if not DEBUG:
        cursor.execute('SELECT v.VideoTitle, c.ChannelName FROM Videos v JOIN Channels c ON (c.ChannelID = v.ChannelID) WHERE v.VideoID IN (SELECT VideoID FROM Tags t JOIN Describes d ON (d.TagsID=t.TagsID) WHERE t.Name LIKE "funny") AND v.Likes > 100 ORDER BY v.Views DESC LIMIT 15;')
        result = cursor.fetchall()
    else:
        result = [('8559102348', '[None]'), ('2629037028', 'funny'), ('1314675381', 'minecraft'), ('1304252895', 'challenge'), ('1216853513', 'BTS')]

    return result

@app.route('/del',methods = ['POST'])
def delete_data():
    if not DEBUG:
        try:
            cursor.execute("DELETE FROM Tags WHERE TagsID="+str(request.json["ID"]+";"))
            connection.commit()
        except Error as e:
            print(f"error '{e}'")
            return jsonify(success=False)
    else:
        print("Delete",request.json["ID"])
    return jsonify(success=True)  
        
@app.route('/insert',methods = ['POST'])
def insert_data():
    if not DEBUG:
        try:
            cursor.execute("INSERT INTO Tags(TagsID,Name,NumVid,NumChannel) VALUES(%s,'%s',%s,%s);" % tuple(request.json.values()))
            connection.commit()
        except Error as e:
            print(f"error '{e}'")
            return jsonify(success=False)
    else:
        print("Insert",request.json.values())
    return jsonify(success=True)  
        
@app.route('/search',methods = ['POST'])
def search_data():
    if request.json["op"]=="title":
        if not DEBUG:
            try:
                cursor.execute("SELECT VideoTitle,Region,Views FROM Videos WHERE VideoTitle LIKE '%"+request.json["keyword"]+"%' LIMIT 15;")
                result = cursor.fetchall()
            except Error as e:
                print(f"error '{e}'")
                
        else:
            print("Search for vid title",request.json["keyword"])
            result = ["Test","Teest",123]
    else:
        if not DEBUG:
            try:
                cursor.execute("SELECT TagsID,Name,NumVid,NumChannel FROM Tags WHERE Name LIKE '%"+request.json["keyword"]+"%' LIMIT 15;")
                result = cursor.fetchall()
            except Error as e:
                print(f"error '{e}'")
        else:
            print("Search for Tag name",request.json["keyword"])
            result = ["456","Test",555,123]
    return result

@app.route('/update',methods = ['POST'])
def update_data():
    if not DEBUG:
        try:
            cursor.execute("UPDATE Tags SET Name='%s',NumVid=%s,NumChannel=%s WHERE TagsID=%s;" % tuple([request.json["TagsName"],request.json["Numvid"],request.json["NumChannel"],request.json["TagsID"]]))
            connection.commit()
        except Error as e:
            print(f"error '{e}'")
            return jsonify(success=False)
    else:
        print("update",request.json.values())
    return jsonify(success=True)  

if __name__ == '__main__':
    if not DEBUG:
        print("Trying to connect to database...")
        try:
            connection = mysql.connector.connect(host="34.70.1.192", user="root", passwd="123456",database='testdatabase')
        except Error as e:
            print(f"error '{e}'")
        
        print("connected")
        cursor = connection.cursor()
    app.run(host='0.0.0.0',port=5000,debug = False)
    if not DEBUG:
        connection.close()
    
    