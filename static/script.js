function sendQuery() {
    let query = document.getElementById("query").value;
    let chatbox = document.getElementById("chatbox");

    fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        let response = document.createElement("p");
        response.innerHTML = "<strong>Bot:</strong> " + JSON.stringify(data);
        chatbox.appendChild(response);
    });
}
