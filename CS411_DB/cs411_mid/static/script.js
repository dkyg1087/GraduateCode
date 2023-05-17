window.post = async function(url,data) {
    return await fetch(url, {method: "POST", headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
}

window.get = async function(url){
    return await fetch(url,{method: "GET", headers: {'Content-Type': 'application/json'}});
}

function sendQuery(number){
    console.log(number)
    result = get("/advQ"+number.toString()).then(response => response.json()).then(data =>{
        console.log(data);
        document.getElementById("output_AQ").innerHTML = JSON.stringify(data);
    });
}

function delete_data(){
    TagID = document.getElementById("del").value;
    post("/del",{ID:TagID});
    console.log("%s deleted",TagID);
}

function insert_data(){
    TagsID = document.getElementById("TagsID_I").value;
    TagsName = document.getElementById("Name_I").value;
    Numvid = document.getElementById("NumVid_I").value;
    NumChannel = document.getElementById("NumChannel_I").value;
    post("/insert",{TagsID:TagsID,TagsName:TagsName,Numvid:Numvid,NumChannel:NumChannel});
    console.log("data inserted");
}

function search_data(targ){
    if (targ == 'vid'){
        title = document.getElementById("Video_S").value;
        result = post("/search",{op:'title',keyword:title}).then(response => response.json()).then(data =>{
            console.log(data);
            document.getElementById("output_S").innerHTML = JSON.stringify(data);
        });
    }
    else{
        name_ = document.getElementById("Tag_S").value;
        result = post("/search",{op:'tag',keyword:name_}).then(response => response.json()).then(data =>{
            console.log(data);
            document.getElementById("output_S").innerHTML = JSON.stringify(data);
        });
    }
        
    
}

function update_data(){
    TagsID = document.getElementById("TagsID_U").value;
    TagsName = document.getElementById("Name_U").value;
    Numvid = document.getElementById("NumVid_U").value;
    NumChannel = document.getElementById("NumChannel_U").value;
    post("/update",{TagsID:TagsID,TagsName:TagsName,Numvid:Numvid,NumChannel:NumChannel});
    console.log("data updated");
}