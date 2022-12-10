window.post = function(url, data) {
    return fetch(url, {method: "POST", headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
}
function setEmail() {
    mail1 = document.getElementById("mail").value;
    mail = mail1
    document.getElementById("mail").value = "";
    post("http://172.16.109.23:5000/email", {action: "set",email:mail});
}
function delEmail(){
    post("http://172.16.109.23:5000/email", {action: "del"});
}

function left(){
    post("http://172.16.109.23:5000/control", {control: "left"});
}

function right(){
    post("http://172.16.109.23:5000/control", {control: "right"});
}

document.addEventListener("keydown", function(event) {
    if (document.activeElement == document.getElementById("mail")) {
        return
    }
    if (event.key == "ArrowLeft" || event.key=="a") {
        document.getElementById("left").style.borderColor = "green"
        left()
    }
    else if (event.key == "ArrowRight" || event.key=="d") {
        document.getElementById("right").style.borderColor = "green"
        right()
    }
  });

document.addEventListener("keyup", function(event) {
    if (event.key == "ArrowLeft" || event.key=="a") {
        document.getElementById("left").style.borderColor = "black"
    }
    else if (event.key == "ArrowRight" || event.key=="d") {
        document.getElementById("right").style.borderColor = "black"
    }
});

function down(targ){
    if(targ == "left"){
        document.getElementById("left").style.borderColor = "green"
        left()
    }
    else if (targ == "right"){
        document.getElementById("right").style.borderColor = "green"
        right()
    }
}

function resume(targ){
    if(targ == "left"){
        document.getElementById("left").style.borderColor = "black"
    }
    else if (targ == "right"){
        document.getElementById("right").style.borderColor = "black"
    }
}