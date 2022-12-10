from flask import Flask, render_template, request,jsonify
import smtplib
import os
from email.message import EmailMessage
import ssl
import time


STIME = 0

FC = True
if FC:
    import picar_4wd as fc
    fc.servo.set_angle(0)

app = Flask(__name__)

angle = 0
recv_email = ""

@app.route("/")
def index():
   return render_template("index.html")

@app.route("/email",methods=["POST"])
def email():
    global recv_email,STIME
    if request.json['action']=="set":
        recv_email = request.json['email']
        STIME = 0
        print("Email set",recv_email)
    elif request.json['action']=="del":
        recv_email = ""
        print("Email del")
    elif request.json['action']=="send":
        c_time = request.json['time']
        obj = request.json['object'] 
        if recv_email == "" or time.time()-STIME < 300:
            print("email was empty or time frame.",time.time()-STIME)
            return jsonify(success=False)
        sender = "urcamera123@gmail.com"
        email_password = os.environ.get("EMAILPWD")
        
        subject = "Object Detected on Camera"
        body = "At " + c_time + ", the camera has detected a " + obj +" in the scene.\n If that isn't something you expected. Please be aware.\n Check out the camera at 172.16.109.23:5000 ."
        
        em = EmailMessage()
        em['From']=sender
        em['To']=recv_email
        em['Subject']=subject
        em.set_content(body)
        
        context = ssl.create_default_context()
        
        with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
            smtp.login(sender,email_password)
            smtp.sendmail(sender,recv_email,em.as_string())
        STIME = time.time()
        print("Email send")
    return jsonify(success=True)         

@app.route("/control",methods=["POST"])
def control():
    global angle
    if not FC:
        print("Test")
        return jsonify(success=True)
    if request.json["control"] == "left":
        if angle >= 90:
            angle = 90
        else:
            angle+=5
    else:
        if angle <= -90:
            angle = -90
        else:
            angle-=5
    fc.servo.set_angle(angle)
    return jsonify(success=True)  
        
if __name__ == '__main__':
   app.run(debug = True)
   app.run(host='0.0.0.0')