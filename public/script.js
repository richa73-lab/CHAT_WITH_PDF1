const uploadForm = document.getElementById("upload-form");
const pdfInput = document.getElementById("pdf-file");
const uploadStatus = document.getElementById("upload-status");

const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatWindow = document.getElementById("chat-window");

let pdfLoaded = false;

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = pdfInput.files[0];
  if (!file) return;

  uploadStatus.textContent = "Uploading & indexing PDF...";
  const formData = new FormData();
  formData.append("pdf", file);

  try {
    const res = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (res.ok) {
      uploadStatus.textContent = "✅ PDF indexed successfully!";
      pdfLoaded = true;
    } else {
      uploadStatus.textContent = "❌ Error: " + data.error;
    }
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = "❌ Error uploading file.";
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;
  if (!pdfLoaded) {
    alert("Please upload a PDF first.");
    return;
  }

  addMessage("user", message);
  userInput.value = "";

  addMessage("bot", "Thinking...");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: message }),
    });

    const data = await res.json();
    chatWindow.removeChild(chatWindow.lastChild); // remove "Thinking..."

    if (res.ok) {
      addMessage("bot", data.answer);
    } else {
      addMessage("bot", "Error: " + data.error);
    }
  } catch (err) {
    console.error(err);
    chatWindow.removeChild(chatWindow.lastChild);
    addMessage("bot", "Error talking to server.");
  }
});
function addMessage(sender, text) {
  const msgDiv = document.createElement("div");

  if (sender === "user") {
    msgDiv.className = "chat-msg user-msg";
  } else {
    msgDiv.className = "chat-msg bot-msg";
  }

  msgDiv.textContent = text;
  chatWindow.appendChild(msgDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}
