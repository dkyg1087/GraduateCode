var socket
var interval
window.addEventListener('load', (event) => {
    socket = new WebSocket('ws://172.16.109.23:8564');
    socket.addEventListener('open', (event) => {
        socket.send('#Hello Server!');
    });
    socket.addEventListener('message', (event) => {
        let str = event.data
        const myArr = str.split(',')
        document.getElementById("Speed").innerText = myArr[0]
        document.getElementById("Distance").innerText = myArr[1]
        document.getElementById("Temperature").innerText = myArr[2]
        document.getElementById("Battery").innerText = myArr[3]
    });
    socket.addEventListener('error', (event) => {
        let reload = confirm(' Socket connection failed,do you want to reload?')
        if (reload){
            window.location.reload()
        }
    })
    socket.addEventListener('close', () => {
        clearInterval(interval)
        let reload = confirm(' Socket connection failed,do you want to reload?')
        if (reload){
            window.location.reload()
        }
    });

    interval = setInterval(function(){
        if (socket.readyState == 1){
            socket.send("query");
        }
    }, 50);
  });

function message_send(){
    socket.send("#"+document.getElementById('input').value)
    document.getElementById('input').value = ""
}

function send_cmd(str){
    socket.send(str)
}

//socket.send('Hello Server!');
//   socket.send('Hello Server!');