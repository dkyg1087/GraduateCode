window.post = async function(url,data) {
    return await fetch(url, {method: "POST", headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
}

function send(q){

    console.log("Send,",q);
    result = post("/query",{query:q}).then(response => response.json()).then(data =>{

        if (!data["result"]){
            alert("There was a error in your query. " +data["msg"])
        }
        else{
            console.log(data["msg"]);
            console.log(typeof data["msg"]);
            res = data["msg"]
            var array1 = res.map(item => item[0]);
            var array2 = res.map(item => item[1]);
            show_chart(array1,array2)
        }
        
    });
}
