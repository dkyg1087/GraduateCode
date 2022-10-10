document.addEventListener("keydown", function(event) {
    if (document.activeElement == document.getElementById("input")) {
        return
    }
    if (event.key == "ArrowLeft" || event.key=="a") {
        document.getElementById("text").innerHTML = "left"
        document.getElementById("left").style.borderColor = "green"
        send_cmd("left")
    }
    else if (event.key == "ArrowUp" || event.key=="w") {
        document.getElementById("text").innerHTML = "up"
        document.getElementById("up").style.borderColor = "green"
        send_cmd("forward")
    }
    else if (event.key == "ArrowDown" || event.key=="s") {
        document.getElementById("text").innerHTML = "down"
        document.getElementById("down").style.borderColor = "green"
        send_cmd("backward")
    }
    else if (event.key == "ArrowRight" || event.key=="d") {
        document.getElementById("text").innerHTML = "right"
        document.getElementById("right").style.borderColor = "green"
        send_cmd("right")
    }
  });

document.addEventListener("keyup", function(event) {
    if (document.activeElement == document.getElementById("input")) {
        if (event.key == "Enter") {
            message_send()
        }
    }
    if (event.key == "ArrowLeft" || event.key=="a") {
        document.getElementById("text").innerHTML = "stop"
        document.getElementById("left").style.borderColor = "black"
        send_cmd("stop")
    }
    else if (event.key == "ArrowUp" || event.key=="w") {
        document.getElementById("text").innerHTML = "stop"
        document.getElementById("up").style.borderColor = "black"
        send_cmd("stop")
    }
    else if (event.key == "ArrowDown" || event.key=="s") {
        document.getElementById("text").innerHTML = "stop"
        document.getElementById("down").style.borderColor = "black"
        send_cmd("stop")
    }
    else if (event.key == "ArrowRight" || event.key=="d") {
        document.getElementById("text").innerHTML = "stop"
        document.getElementById("right").style.borderColor = "black"
        send_cmd("stop")
    }
});
