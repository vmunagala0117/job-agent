/* ================================================================
   Job Agent — Chat Application Logic
   ================================================================ */

(() => {
    "use strict";

    // --- Configuration ---
    const API_BASE = "";  // Same origin

    // --- DOM refs ---
    const app           = document.querySelector(".app");
    const chatContainer = document.getElementById("chat-container");
    const messagesDiv   = document.getElementById("messages");
    const welcomeDiv    = document.getElementById("welcome");
    const typingDiv     = document.getElementById("typing");
    const chatForm      = document.getElementById("chat-form");
    const messageInput  = document.getElementById("message-input");
    const sendBtn       = document.getElementById("send-btn");
    const newChatBtn    = document.getElementById("new-chat-btn");

    // Upload elements
    const uploadBtn     = document.getElementById("upload-btn");
    const fileInput     = document.getElementById("file-input");

    // Trace panel elements
    const tracesBtn     = document.getElementById("traces-btn");
    const tracePanel    = document.getElementById("trace-panel");
    const traceEntries  = document.getElementById("trace-entries");
    const traceClearBtn = document.getElementById("trace-clear-btn");

    // Resume badge
    const resumeBadge   = document.getElementById("resume-badge");
    const resumeName    = document.getElementById("resume-name");

    // Profile selector
    const profileSelect = document.getElementById("profile-select");

    // --- State ---
    let sessionId = null;
    let isProcessing = false;

    // --- Markdown setup ---
    marked.setOptions({
        highlight: (code, lang) => {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true,
    });

    // --- Utility ---
    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
    }

    function autoResize() {
        messageInput.style.height = "auto";
        messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
    }

    function setProcessing(active) {
        isProcessing = active;
        sendBtn.disabled = active || !messageInput.value.trim();
        messageInput.disabled = active;
        typingDiv.style.display = active ? "block" : "none";
        if (!active) {
            messageInput.focus();
        }
    }

    // --- Message Rendering ---
    function addMessage(role, text) {
        // Hide welcome screen on first message
        if (welcomeDiv) {
            welcomeDiv.style.display = "none";
        }

        const msgEl = document.createElement("div");
        msgEl.className = `message ${role}`;

        const avatarEl = document.createElement("div");
        avatarEl.className = "message-avatar";

        if (role === "user") {
            avatarEl.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>`;
        } else {
            avatarEl.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="2" y="7" width="20" height="14" rx="2" ry="2"/>
                    <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>
                </svg>`;
        }

        const contentEl = document.createElement("div");
        contentEl.className = "message-content";

        if (role === "user") {
            contentEl.textContent = text;
        } else if (role === "error") {
            contentEl.textContent = text;
        } else {
            // Render markdown for assistant messages
            contentEl.innerHTML = marked.parse(text);
        }

        msgEl.appendChild(avatarEl);
        msgEl.appendChild(contentEl);
        messagesDiv.appendChild(msgEl);
        scrollToBottom();

        return msgEl;
    }

    // --- Trace Rendering ---
    function classifyTrace(message) {
        const msg = message.toLowerCase();
        if (msg.includes("error") || msg.includes("exception") || msg.includes("failed")) return "trace-error";
        if (msg.includes("warning") || msg.includes("warn")) return "trace-warning";
        if (msg.includes("tool") || msg.includes("function") || msg.includes("calling")) return "trace-tool";
        if (msg.includes("agent") || msg.includes("classifier") || msg.includes("routing")) return "trace-agent";
        return "trace-info";
    }

    function renderTraces(traces) {
        if (!traces || traces.length === 0) return;

        // Remove "empty" placeholder
        const emptyEl = traceEntries.querySelector(".trace-empty");
        if (emptyEl) emptyEl.remove();

        // Add separator for this batch
        const sep = document.createElement("div");
        sep.className = "trace-entry";
        sep.style.borderBottom = "1px solid #334155";
        sep.style.paddingBottom = "6px";
        sep.style.marginBottom = "4px";
        sep.innerHTML = `<span class="trace-time">───</span> response`;
        traceEntries.appendChild(sep);

        traces.forEach((t) => {
            const el = document.createElement("div");
            const cls = classifyTrace(t.message);
            el.className = `trace-entry ${cls}`;

            const time = new Date(t.timestamp * 1000).toLocaleTimeString([], {
                hour: "2-digit", minute: "2-digit", second: "2-digit",
            });

            el.innerHTML = `<span class="trace-time">${time}</span>${escapeHtml(t.message)}`;
            traceEntries.appendChild(el);
        });

        traceEntries.scrollTop = traceEntries.scrollHeight;
    }

    function clearTraces() {
        traceEntries.innerHTML = `<div class="trace-empty">No traces yet. Send a message to see agent activity.</div>`;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // --- API calls ---
    async function sendMessage(text) {
        addMessage("user", text);
        setProcessing(true);
        scrollToBottom();

        try {
            const res = await fetch(`${API_BASE}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    session_id: sessionId,
                }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            sessionId = data.session_id;
            addMessage("assistant", data.response);

            // Render traces if available
            if (data.traces && data.traces.length > 0) {
                renderTraces(data.traces);
            }

        } catch (err) {
            console.error("Chat error:", err);
            addMessage("error", `Something went wrong: ${err.message}`);
        } finally {
            setProcessing(false);
        }
    }

    async function resetChat() {
        try {
            await fetch(`${API_BASE}/api/chat/reset`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId }),
            });
        } catch (e) {
            // Ignore — we'll create a new session on next message
        }

        sessionId = null;
        messagesDiv.innerHTML = "";
        if (welcomeDiv) {
            welcomeDiv.style.display = "flex";
        }
        messageInput.value = "";
        autoResize();
        sendBtn.disabled = true;

        // Clear trace panel
        clearTraces();

        // Hide resume badge
        resumeBadge.hidden = true;
        resumeName.textContent = "";

        messageInput.focus();
    }

    // --- Resume Upload ---
    async function uploadResume(file) {
        const formData = new FormData();
        formData.append("file", file);
        if (sessionId) {
            formData.append("session_id", sessionId);
        }

        uploadBtn.classList.add("uploading");

        try {
            const res = await fetch(`${API_BASE}/api/upload-resume`, {
                method: "POST",
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            sessionId = data.session_id;

            // Show resume badge
            resumeBadge.hidden = false;
            resumeName.textContent = data.filename;

            // Show summary as assistant message
            addMessage("assistant", data.summary);

            // Refresh profile dropdown to include the new profile
            loadProfiles();

        } catch (err) {
            console.error("Upload error:", err);
            addMessage("error", `Failed to upload resume: ${err.message}`);
        } finally {
            uploadBtn.classList.remove("uploading");
            fileInput.value = "";  // Reset so same file can be re-uploaded
        }
    }

    // --- Event Listeners ---

    // Form submission
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = messageInput.value.trim();
        if (!text || isProcessing) return;
        messageInput.value = "";
        autoResize();
        sendBtn.disabled = true;
        sendMessage(text);
    });

    // Textarea: auto-resize + enable/disable send
    messageInput.addEventListener("input", () => {
        autoResize();
        sendBtn.disabled = isProcessing || !messageInput.value.trim();
    });

    // Enter to send, Shift+Enter for newline
    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // New chat button
    newChatBtn.addEventListener("click", resetChat);

    // Quick action chips
    document.querySelectorAll(".chip[data-message]").forEach((chip) => {
        chip.addEventListener("click", () => {
            const msg = chip.getAttribute("data-message");
            if (msg && !isProcessing) {
                sendMessage(msg);
            }
        });
    });

    // Upload button → trigger file input
    uploadBtn.addEventListener("click", () => {
        if (!isProcessing) fileInput.click();
    });

    // File selected → upload
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) uploadResume(file);
    });

    // Trace panel toggle
    tracesBtn.addEventListener("click", () => {
        app.classList.toggle("traces-open");
        tracesBtn.classList.toggle("active");
    });

    // Trace clear button
    traceClearBtn.addEventListener("click", clearTraces);

    // --- Profile Selector ---
    async function loadProfiles() {
        try {
            const res = await fetch(`${API_BASE}/api/profiles`);
            if (!res.ok) return;
            const profiles = await res.json();

            // Clear existing options (keep the first "No profile" option)
            profileSelect.innerHTML = `<option value="">No profile loaded</option>`;

            if (profiles.length === 0) return;

            profiles.forEach((p) => {
                const opt = document.createElement("option");
                opt.value = p.id;
                const label = p.name || "Unnamed";
                const extra = p.title ? ` — ${p.title}` : "";
                opt.textContent = `${label}${extra} (${p.skills_count} skills)`;
                if (p.is_active) {
                    opt.selected = true;
                    // Also show the resume badge for the active profile
                    resumeBadge.hidden = false;
                    resumeName.textContent = label;
                }
                profileSelect.appendChild(opt);
            });
        } catch (e) {
            console.warn("Could not load profiles:", e);
        }
    }

    profileSelect.addEventListener("change", async () => {
        const id = profileSelect.value;
        if (!id) return;

        try {
            const res = await fetch(`${API_BASE}/api/profiles/select?profile_id=${encodeURIComponent(id)}`, {
                method: "POST",
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            // Update badge
            resumeBadge.hidden = false;
            resumeName.textContent = data.name;

            // Notify in chat
            addMessage("assistant", `Switched to profile: **${data.name}** — ${data.title || "no title"}\nSkills: ${data.skills.join(", ")}`);
        } catch (e) {
            console.error("Profile select error:", e);
            addMessage("error", `Failed to switch profile: ${e.message}`);
        }
    });

    // Load profiles on startup
    loadProfiles();

    // Focus input on page load
    messageInput.focus();
})();
