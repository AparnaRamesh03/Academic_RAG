const API_URL = "http://localhost:8000/ask";

// --- DOM Elements ---
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");
const citationsList = document.getElementById("citations-list");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat");

// --- State ---
let isProcessing = false;

// --- Helper Functions ---

/**
 * Add a message bubble to the chat container
 */
function addMessage(text, role) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    // Convert newlines to breaks for simple formatting
    const formattedText = text.replace(/\n/g, "<br>");

    messageDiv.innerHTML = `
        <div class="message-content">
            ${formattedText}
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Show/Hide typing indicator
 */
function showTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "message assistant typing-indicator";
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingDiv;
}

/**
 * Update the citations sidebar with retrieved chunks
 */
function updateCitations(context) {
    if (!context || context.length === 0) {
        citationsList.innerHTML = `
            <div class="empty-state">
                <p>No specific citations for this response.</p>
            </div>
        `;
        return;
    }

    citationsList.innerHTML = ""; // Clear current list

    // Replace newlines with breaks for better formatting
    context.forEach((text, index) => {
        const formattedText = text.replace(/\n/g, "<br>");
        const card = document.createElement("div");
        card.className = "citation-card";

        // Note: In the minimal brain, we only send the 'text'. 
        // In the full brain, we'll have metadata like page numbers.
        card.innerHTML = `
            <div class="meta">
                <span>Source Chunk #${index + 1}</span>
                <span>(Click to expand)</span>
            </div>
            <div class="excerpt">${formattedText}</div>
        `;

        card.addEventListener("click", () => {
            card.classList.toggle("expanded");
        });

        citationsList.appendChild(card);
    });
}

/**
 * Handle form submission
 */
async function handleSubmit(e) {
    e.preventDefault();

    const query = userInput.value.trim();
    if (!query || isProcessing) return;

    // 1. Update UI for user message
    addMessage(query, "user");
    userInput.value = "";
    setProcessing(true);

    // 2. Add typing indicator
    const indicator = showTypingIndicator();

    try {
        // 3. Fetch from Brain API
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });

        if (!response.ok) throw new Error("Brain API connection failed");

        const data = await response.json();

        // 4. Remove indicator and add assistant message
        indicator.remove();
        addMessage(data.answer, "assistant");

        // 5. Update Citations
        updateCitations(data.context_used);

    } catch (error) {
        console.error("Error:", error);
        indicator.remove();
        addMessage("Sorry, I encountered an error connecting to the Brain. Please ensure it's running on port 8000.", "assistant");
    } finally {
        setProcessing(false);
    }
}

function setProcessing(bool) {
    isProcessing = bool;
    sendBtn.disabled = bool;
    userInput.disabled = bool;
}

// --- Event Listeners ---
chatForm.addEventListener("submit", handleSubmit);

clearChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = "";
    citationsList.innerHTML = `
        <div class="empty-state">
            <p>Ask a question to see source citations here.</p>
        </div>
    `;
    addMessage("Chat cleared. How else can I help you?", "assistant");
});

// Initial Focus
userInput.focus();
