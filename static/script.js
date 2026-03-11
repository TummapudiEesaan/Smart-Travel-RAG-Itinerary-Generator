/**
 * Smart-Travel RAG Itinerary Generator — Frontend Script
 * Handles chat interaction, API communication, and message rendering.
 */

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════

let isLoading = false;

// ═══════════════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════════════

const chatContainer = document.getElementById("chatContainer");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const statusBadge = document.getElementById("statusBadge");

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    // Focus input
    queryInput.focus();

    // Enter key handler
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    // Check API status
    checkStatus();
});

/**
 * Check the API status and update the badge.
 */
async function checkStatus() {
    try {
        const res = await fetch("/api/info");
        const data = await res.json();

        const dot = statusBadge.querySelector(".badge-dot");
        const text = statusBadge.querySelector(".badge-text");

        if (data.api_available) {
            dot.classList.add("connected");
            text.textContent = `${data.entries} entries · AI Ready`;
        } else {
            dot.style.background = "#f59e0b";
            text.textContent = `${data.entries} entries · Offline`;
        }
    } catch {
        const text = statusBadge.querySelector(".badge-text");
        text.textContent = "Disconnected";
    }
}

// ═══════════════════════════════════════════════════════════════
// Message Handling
// ═══════════════════════════════════════════════════════════════

/**
 * Send the user's query to the API.
 */
async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || isLoading) return;

    // Hide welcome message
    const welcome = document.getElementById("welcomeMessage");
    if (welcome) {
        welcome.style.display = "none";
    }

    // Add user message
    addMessage(query, "user");
    queryInput.value = "";
    setLoading(true);

    // Add loading indicator
    const loadingId = addLoadingMessage();

    try {
        const res = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });

        const data = await res.json();

        // Remove loading indicator
        removeMessage(loadingId);

        if (res.ok) {
            addBotMessage(data.response, data.retrieved_count);
        } else {
            addErrorMessage(data.error || "Something went wrong. Please try again.");
        }
    } catch (err) {
        removeMessage(loadingId);
        addErrorMessage("Failed to connect to the server. Make sure the server is running.");
    } finally {
        setLoading(false);
        queryInput.focus();
    }
}

/**
 * Send a suggestion chip query.
 */
function sendSuggestion(chipEl) {
    queryInput.value = chipEl.textContent;
    sendQuery();
}

/**
 * Add a user message to the chat.
 */
function addMessage(text, type) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = type === "user" ? "✈️" : "🏔️";

    const content = document.createElement("div");
    content.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;

    content.appendChild(bubble);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);

    scrollToBottom();
}

/**
 * Add a bot response message with markdown rendering.
 */
function addBotMessage(text, retrievedCount) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot-message";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "🏔️";

    const content = document.createElement("div");
    content.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    // Add retrieved count badge
    if (retrievedCount > 0) {
        const badge = document.createElement("div");
        badge.className = "retrieved-badge";
        badge.innerHTML = `📚 ${retrievedCount} sources used`;
        bubble.appendChild(badge);
    }

    // Render markdown-like content
    const rendered = document.createElement("div");
    rendered.innerHTML = renderMarkdown(text);
    bubble.appendChild(rendered);

    content.appendChild(bubble);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);

    scrollToBottom();
}

/**
 * Add an error message.
 */
function addErrorMessage(text) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot-message";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "⚠️";

    const content = document.createElement("div");
    content.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble error-bubble";

    const p = document.createElement("p");
    p.textContent = text;
    bubble.appendChild(p);

    content.appendChild(bubble);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);

    scrollToBottom();
}

/**
 * Add a loading indicator message.
 */
function addLoadingMessage() {
    const id = "loading-" + Date.now();

    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot-message";
    messageDiv.id = id;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "🏔️";

    const content = document.createElement("div");
    content.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    const dotsContainer = document.createElement("div");
    dotsContainer.className = "loading-dots";
    dotsContainer.innerHTML = "<span></span><span></span><span></span>";

    const text = document.createElement("p");
    text.style.fontSize = "13px";
    text.style.color = "#94a3b8";
    text.style.marginTop = "8px";
    text.textContent = "Searching knowledge base & generating response...";

    bubble.appendChild(dotsContainer);
    bubble.appendChild(text);
    content.appendChild(bubble);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);

    scrollToBottom();
    return id;
}

/**
 * Remove a message by its ID.
 */
function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) {
        el.style.animation = "messageOut 0.2s ease forwards";
        setTimeout(() => el.remove(), 200);
    }
}

// ═══════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════

/**
 * Simple markdown-to-HTML renderer.
 */
function renderMarkdown(text) {
    if (!text) return "";

    let html = text;

    // Escape HTML
    html = html
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Headers (### before ## before #)
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Horizontal rules
    html = html.replace(/^---$/gm, "<hr>");

    // Unordered lists
    html = html.replace(/^[\-\•] (.+)$/gm, "<li>$1</li>");

    // Numbered lists
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

    // Wrap consecutive <li> tags in <ul>
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Line breaks to paragraphs
    html = html
        .split(/\n\n+/)
        .map((block) => {
            block = block.trim();
            if (!block) return "";
            if (
                block.startsWith("<h") ||
                block.startsWith("<ul") ||
                block.startsWith("<ol") ||
                block.startsWith("<hr")
            ) {
                return block;
            }
            return `<p>${block.replace(/\n/g, "<br>")}</p>`;
        })
        .join("\n");

    return html;
}

/**
 * Scroll chat to the bottom smoothly.
 */
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: "smooth",
        });
    });
}

/**
 * Toggle loading state.
 */
function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    queryInput.disabled = loading;

    if (!loading) {
        queryInput.focus();
    }
}
