// Send user message and display bot response
function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();

    if (message === "") return;

    // Display user message
    addMessage(message, "user-message");
    input.value = "";

    // Send to Flask backend
    fetch("/ask", {
        method: "POST",
        body: new URLSearchParams({ msg: message }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
    })
    .then(response => response.json())
    .then(data => {
        addMessage(data.response, "bot-message");
    })
    .catch(error => {
        addMessage("⚠️ Error: Could not connect to server.", "bot-message");
    });
}

// Function to add messages to chat box
function addMessage(text, className) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("div");
    messageElement.className = "message " + className;
    messageElement.innerText = text;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // auto-scroll
}
