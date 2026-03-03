/* ================================================================
   Job Agent — Chat Application Logic
   ================================================================ */

(() => {
    "use strict";

    // --- Configuration ---
    const API_BASE = "";  // Same origin

    // Helper to build absolute API URLs. If `API_BASE` is empty, use window.location.origin.
    function apiUrl(path) {
        if (!path.startsWith('/')) path = '/' + path;
        if (API_BASE && API_BASE.length > 0) {
            // Strip possible trailing slash from API_BASE
            const base = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE;
            return base + path;
        }
        return new URL(path, window.location.origin).href;
    }

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
    const packagesBtn   = document.getElementById("packages-btn");

    // Resume badge
    const resumeBadge   = document.getElementById("resume-badge");
    const resumeName    = document.getElementById("resume-name");

    // Profile selector
    const profileSelect = document.getElementById("profile-select");
    const profileEditBtn = document.getElementById("profile-edit-btn");
    const profileNewBtn  = document.getElementById("profile-new-btn");
    const packagesPanel = document.getElementById("packages-panel");
    const packagesRefreshBtn = document.getElementById("packages-refresh-btn");
    const packagesRunSelect = document.getElementById("packages-run-select");
    const packagesList = document.getElementById("packages-list");
    const packagesDownloadRunBtn = document.getElementById("packages-download-run");
    const packagesDownloadSelectedBtn = document.getElementById("packages-download-selected");
    const packagesCloseBtn = document.getElementById("packages-close-btn");
    const packagesBackdrop = document.getElementById("packages-backdrop");

    // Profile modal elements
    const profileModal      = document.getElementById("profile-modal");
    const profileForm       = document.getElementById("profile-form");
    const modalCloseBtn     = document.getElementById("profile-modal-close");
    const modalCancelBtn    = document.getElementById("profile-modal-cancel");
    const pfSaveBtn         = document.getElementById("pf-save-btn");
    const pfResumeBtn       = document.getElementById("pf-resume-btn");
    const pfResumeInput     = document.getElementById("pf-resume");
    const pfResumeName      = document.getElementById("pf-resume-name");

    // --- State ---
    let sessionId = null;
    let isProcessing = false;
    let assistantMsgIndex = 0;  // Tracks assistant message index for feedback

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
    function addMessage(role, text, metadata) {
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

            // Convert plain URLs in the rendered content into clickable anchors
            (function convertPlainUrlsToAnchors(el) {
                const urlRegex = /https?:\/\/[\w\-._~:\/?#\[\]@!$&'()*+,;=%]+/g;

                // Collect text nodes first (TreeWalker can be disrupted when we replace nodes)
                const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
                const textNodes = [];
                while (walker.nextNode()) textNodes.push(walker.currentNode);

                textNodes.forEach(node => {
                    const txt = node.nodeValue;
                    if (!txt) return;
                    let match;
                    urlRegex.lastIndex = 0;
                    if ((match = urlRegex.exec(txt)) !== null) {
                        const frag = document.createDocumentFragment();
                        let lastIndex = 0;
                        urlRegex.lastIndex = 0;
                        while ((match = urlRegex.exec(txt)) !== null) {
                            const url = match[0];
                            const before = txt.slice(lastIndex, match.index);
                            if (before) frag.appendChild(document.createTextNode(before));
                            const a = document.createElement('a');
                            a.href = url;
                            a.textContent = url;
                            a.target = '_blank';
                            a.rel = 'noopener noreferrer';
                            a.className = 'assistant-link';
                            frag.appendChild(a);
                            lastIndex = urlRegex.lastIndex;
                        }
                        const after = txt.slice(lastIndex);
                        if (after) frag.appendChild(document.createTextNode(after));
                        node.parentNode.replaceChild(frag, node);
                    }
                });

                // Ensure existing anchors open safely in a new tab
                const anchors = el.querySelectorAll('a');
                anchors.forEach(a => {
                    if (!a.target) a.target = '_blank';
                    a.rel = 'noopener noreferrer';
                });
            })(contentEl);

            // Add metadata bar + feedback buttons for assistant messages
            const currentMsgIdx = assistantMsgIndex++;
            const metaBar = document.createElement("div");
            metaBar.className = "message-meta";

            // Metadata items
            if (metadata) {
                if (metadata.agent) {
                    const agentLabel = metadata.agent === "job_search_agent" ? "Job Search" : "App Prep";
                    metaBar.innerHTML += `<span class="meta-item" title="Routed agent">${agentLabel}</span>`;
                }
                if (metadata.classifier_confidence != null) {
                    metaBar.innerHTML += `<span class="meta-item" title="Classifier confidence">${metadata.classifier_confidence}%</span>`;
                }
                if (metadata.usage) {
                    const u = metadata.usage;
                    metaBar.innerHTML += `<span class="meta-item" title="Token usage">${u.total_tokens} tok</span>`;
                }
                if (metadata.elapsed_ms != null) {
                    const secs = (metadata.elapsed_ms / 1000).toFixed(1);
                    metaBar.innerHTML += `<span class="meta-item" title="Response time">${secs}s</span>`;
                }
                if (metadata.tool_count != null && metadata.tool_count > 0) {
                    metaBar.innerHTML += `<span class="meta-item" title="Tools used">${metadata.tool_count} tools</span>`;
                }
            }

            // Feedback buttons
            const fbDiv = document.createElement("div");
            fbDiv.className = "feedback-btns";
            fbDiv.innerHTML = `
                <button class="fb-up" title="Good response" data-idx="${currentMsgIdx}">&#x1F44D;</button>
                <button class="fb-down" title="Poor response" data-idx="${currentMsgIdx}">&#x1F44E;</button>
            `;
            metaBar.appendChild(fbDiv);
            contentEl.appendChild(metaBar);

            // Attach feedback click handlers
            const upBtn = fbDiv.querySelector(".fb-up");
            const downBtn = fbDiv.querySelector(".fb-down");
            upBtn.addEventListener("click", () => sendFeedback(currentMsgIdx, "up", upBtn, downBtn));
            downBtn.addEventListener("click", () => sendFeedback(currentMsgIdx, "down", upBtn, downBtn));
        }

        msgEl.appendChild(avatarEl);
        msgEl.appendChild(contentEl);
        messagesDiv.appendChild(msgEl);
        scrollToBottom();

        return msgEl;
    }

    // --- Feedback ---
    async function sendFeedback(msgIndex, rating, upBtn, downBtn) {
        // Disable both buttons
        upBtn.classList.add("disabled");
        downBtn.classList.add("disabled");

        // Highlight the selected one
        if (rating === "up") {
            upBtn.classList.add("selected-up");
        } else {
            downBtn.classList.add("selected-down");
        }

        try {
            await fetch(`${API_BASE}/api/feedback`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    message_index: msgIndex,
                    rating: rating,
                }),
            });
        } catch (e) {
            console.warn("Feedback submission failed:", e);
        }
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

            // Build metadata object for the response bar
            const meta = {
                usage: data.usage || null,
                classifier_confidence: data.classifier_confidence,
                elapsed_ms: data.elapsed_ms,
                agent: data.agent || null,
                tool_count: data.tool_count,
            };
            addMessage("assistant", data.response, meta);

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
        assistantMsgIndex = 0;
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

    // Trace panel toggle (with mutual exclusion)
    tracesBtn.addEventListener("click", () => {
        const willOpen = !app.classList.contains("traces-open");
        // Close packages panel first if open
        if (willOpen) closePackagesPanel();
        app.classList.toggle("traces-open");
        tracesBtn.classList.toggle("active");
    });

    // Note: packages panel toggles later in the file (behaves like Traces).

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

    // --- Packages UI ---
    async function loadRunsIntoSelect() {
        try {
            const res = await fetch(apiUrl('/api/cron/runs'));
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const runs = await res.json();
            packagesRunSelect.innerHTML = "";
            if (!runs || runs.length === 0) {
                const opt = document.createElement('option');
                opt.value = "";
                opt.textContent = "No runs found";
                packagesRunSelect.appendChild(opt);
                packagesList.innerHTML = `<div class="packages-empty">No packages available.</div>`;
                return;
            }
            runs.forEach((r) => {
                const opt = document.createElement('option');
                opt.value = r.id;
                const label = `${r.profile_name || 'run'} — ${new Date(r.created_at).toLocaleString()}`;
                opt.textContent = label;
                packagesRunSelect.appendChild(opt);
            });
            // load packages for the first run
            if (packagesRunSelect.options.length > 0) {
                packagesRunSelect.selectedIndex = 0;
                loadPackagesForRun(packagesRunSelect.value);
            }
        } catch (e) {
            console.warn('Could not load runs:', e);
            packagesList.innerHTML = `<div class="packages-empty">Failed to load runs.</div>`;
        }
    }

    async function loadPackagesForRun(runId) {
        if (!runId) return;
        packagesList.innerHTML = `<div class="packages-loading">Loading packages…</div>`;
        try {
            const res = await fetch(apiUrl(`/api/packages?run_id=${encodeURIComponent(runId)}`));
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const pkgs = await res.json();
            if (!pkgs || pkgs.length === 0) {
                packagesList.innerHTML = `<div class="packages-empty">No packages for this run.</div>`;
                return;
            }
            packagesList.innerHTML = "";
            pkgs.forEach((p) => {
                const el = document.createElement('div');
                el.className = 'pkg-item';
                el.innerHTML = `
                    <div class="pkg-toggle" role="button" tabindex="0">
                        <svg class="pkg-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
                        <label class="pkg-checkbox"><input type="checkbox" data-pkgid="${p.id}"></label>
                        <div class="pkg-info">
                            <div class="pkg-title">${escapeHtml(p.job_title || 'Untitled')}</div>
                            <div class="pkg-meta">${escapeHtml(p.company || '')} — ${p.created_at ? new Date(p.created_at).toLocaleString() : ''}</div>
                        </div>
                    </div>
                    <div class="pkg-body">
                        <div class="pkg-actions">
                            <button class="btn btn-sm btn-primary pkg-preview-inline" data-id="${p.id}">Preview</button>
                            <button class="btn btn-sm pkg-download" data-id="${p.id}">Download</button>
                        </div>
                        <div class="pkg-detail" data-id="${p.id}"></div>
                    </div>
                `;
                packagesList.appendChild(el);
            });

            // Toggle expand/collapse — skip clicks on checkboxes
            packagesList.querySelectorAll('.pkg-toggle').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    // Don't toggle when clicking the checkbox or its label
                    if (e.target.closest('.pkg-checkbox')) return;
                    const item = e.currentTarget.closest('.pkg-item');
                    const wasOpen = item.classList.contains('open');
                    item.classList.toggle('open');
                    // Lazy-load preview on first open
                    if (!wasOpen) {
                        const detailDiv = item.querySelector('.pkg-detail');
                        if (!detailDiv.dataset.loaded) {
                            loadInlinePreview(detailDiv.getAttribute('data-id'), detailDiv);
                        }
                    }
                });
            });

            // Download buttons
            packagesList.querySelectorAll('.pkg-download').forEach(btn => btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const id = e.currentTarget.getAttribute('data-id');
                const url = apiUrl(`/api/packages/${encodeURIComponent(id)}/download?formats=md,txt`);
                window.open(url, '_blank');
            }));

            // Inline preview buttons
            packagesList.querySelectorAll('.pkg-preview-inline').forEach(btn => btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const id = e.currentTarget.getAttribute('data-id');
                const detailDiv = e.currentTarget.closest('.pkg-body').querySelector('.pkg-detail');
                loadInlinePreview(id, detailDiv);
            }));
        } catch (e) {
            console.error('Failed to load packages:', e);
            packagesList.innerHTML = `<div class="packages-empty">Failed to load packages.</div>`;
        }
    }

    async function loadInlinePreview(id, container) {
        container.innerHTML = `<div class="packages-loading" style="padding:8px 0">Loading preview…</div>`;
        try {
            const res = await fetch(apiUrl(`/api/packages/${encodeURIComponent(id)}`));
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const p = await res.json();
            container.dataset.loaded = "1";

            const sections = [];
            if (p.resume_suggestions && p.resume_suggestions.length) {
                sections.push(`<div class="pkg-section"><div class="pkg-section-title">Resume Suggestions</div><div class="pkg-section-content">${escapeHtml(p.resume_suggestions.join('\n'))}</div></div>`);
            }
            if (p.cover_letter) {
                sections.push(`<div class="pkg-section"><div class="pkg-section-title">Cover Letter</div><div class="pkg-section-content">${escapeHtml(p.cover_letter)}</div></div>`);
            }
            if (p.intro_email) {
                sections.push(`<div class="pkg-section"><div class="pkg-section-title">Intro Email</div><div class="pkg-section-content">${escapeHtml(p.intro_email)}</div></div>`);
            }
            container.innerHTML = sections.join('') || '<div class="packages-empty" style="padding:8px 0">No content available.</div>';
        } catch (e) {
            console.error('Preview failed:', e);
            container.innerHTML = `<div class="packages-empty" style="padding:8px 0;color:var(--color-error)">Preview failed: ${escapeHtml(e.message)}</div>`;
        }
    }

    async function previewPackage(id) {
        try {
            const res = await fetch(apiUrl(`/api/packages/${encodeURIComponent(id)}`));
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const p = await res.json();
            const parts = [];
            parts.push(`# ${p.job.title} — ${p.job.company}`);
            parts.push(`Package ID: ${p.id}`);
            parts.push(`\n---\n`);
            parts.push(`## Resume suggestions\n${(p.resume_suggestions||[]).join('\n')}`);
            parts.push(`\n---\n`);
            parts.push(`## Cover letter\n${p.cover_letter || ''}`);
            parts.push(`\n---\n`);
            parts.push(`## Intro email\n${p.intro_email || ''}`);

            addMessage('assistant', parts.join('\n\n'));
        } catch (e) {
            console.error('Preview failed:', e);
            addMessage('error', `Preview failed: ${e.message}`);
        }
    }

    // Packages controls
    if (packagesRefreshBtn) packagesRefreshBtn.addEventListener('click', () => loadRunsIntoSelect());
    if (packagesRunSelect) packagesRunSelect.addEventListener('change', (e) => loadPackagesForRun(e.target.value));
    if (packagesDownloadRunBtn) packagesDownloadRunBtn.addEventListener('click', () => {
        const runId = packagesRunSelect.value;
        if (!runId) { addMessage('error', 'No run selected'); return; }
        const url = apiUrl(`/api/packages/download?run_id=${encodeURIComponent(runId)}&formats=md,txt`);
        window.open(url, '_blank');
    });
    if (packagesDownloadSelectedBtn) packagesDownloadSelectedBtn.addEventListener('click', () => {
        const checks = packagesList.querySelectorAll('input[type="checkbox"][data-pkgid]:checked');
        const ids = Array.from(checks).map(c => c.getAttribute('data-pkgid'));
        if (!ids.length) { addMessage('error', 'No packages selected'); return; }
        const url = apiUrl(`/api/packages/download?package_ids=${encodeURIComponent(ids.join(','))}&formats=md,txt`);
        window.open(url, '_blank');
    });

    // ---- Packages panel open / close helpers ----
    function openPackagesPanel() {
        // Close traces first (mutual exclusion)
        if (app.classList.contains('traces-open')) {
            app.classList.remove('traces-open');
            tracesBtn.classList.remove('active');
        }
        app.classList.add('packages-open');
        packagesBtn.classList.add('active');
        loadRunsIntoSelect();
    }

    function closePackagesPanel() {
        app.classList.remove('packages-open');
        packagesBtn.classList.remove('active');
    }

    function togglePackagesPanel() {
        if (app.classList.contains('packages-open')) {
            closePackagesPanel();
        } else {
            openPackagesPanel();
        }
    }

    // Toggle via header button
    if (packagesBtn) {
        packagesBtn.addEventListener('click', () => togglePackagesPanel());
    }

    // Close via panel X button
    if (packagesCloseBtn) {
        packagesCloseBtn.addEventListener('click', () => closePackagesPanel());
    }

    // Close via click outside the panel (replaces backdrop overlay)
    document.addEventListener('mousedown', (e) => {
        if (!app.classList.contains('packages-open')) return;
        // If clicking inside the panel or on the packages toggle button, do nothing
        if (packagesPanel && packagesPanel.contains(e.target)) return;
        if (packagesBtn && packagesBtn.contains(e.target)) return;
        closePackagesPanel();
    });

    // Close via Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && app.classList.contains('packages-open')) {
            closePackagesPanel();
        }
    });

    // --- Profile Modal Logic ---

    function openProfileModal(profileId) {
        profileForm.reset();
        document.getElementById("pf-id").value = "";
        pfResumeName.textContent = "No file chosen";
        pfResumeName.classList.remove("has-file");
        pfResumeInput.value = "";
        hideResumeSummary();

        if (profileId) {
            document.getElementById("profile-modal-title").textContent = "Edit Profile";
            fetchProfileAndPopulate(profileId);
        } else {
            document.getElementById("profile-modal-title").textContent = "New Profile";
        }

        profileModal.hidden = false;
        document.getElementById("pf-name").focus();
    }

    function hideResumeSummary() {
        document.getElementById("pf-resume-summary").hidden = true;
    }

    function showResumeSummary(profile) {
        const section = document.getElementById("pf-resume-summary");
        document.getElementById("pf-extracted-title").textContent = profile.current_title || "—";
        document.getElementById("pf-extracted-years").textContent =
            profile.years_experience != null ? `${profile.years_experience} years` : "—";
        document.getElementById("pf-extracted-skills").textContent =
            (profile.skills && profile.skills.length) ? profile.skills.join(", ") : "—";
        document.getElementById("pf-extracted-summary").textContent = profile.summary || "—";
        section.hidden = false;
    }

    function closeProfileModal() {
        profileModal.hidden = true;
    }

    async function fetchProfileAndPopulate(profileId) {
        try {
            const res = await fetch(`${API_BASE}/api/profiles/${encodeURIComponent(profileId)}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const p = await res.json();

            document.getElementById("pf-id").value           = p.id || "";
            document.getElementById("pf-name").value         = p.name || "";
            document.getElementById("pf-email").value        = p.email || "";
            document.getElementById("pf-desired").value      = (p.desired_titles || []).join(", ");
            document.getElementById("pf-locations").value    = (p.preferred_locations || []).join(", ");
            document.getElementById("pf-remote").value       = p.remote_preference || "";
            document.getElementById("pf-salary").value       = p.min_salary ?? "";
            document.getElementById("pf-industries").value   = (p.industries || []).join(", ");

            // Show resume-extracted details as read-only summary
            if (p.has_resume || p.skills?.length || p.current_title) {
                showResumeSummary(p);
            }

            if (p.has_resume) {
                pfResumeName.textContent = "Resume on file";
                pfResumeName.classList.add("has-file");
            }
        } catch (e) {
            console.error("Failed to load profile:", e);
            addMessage("error", `Failed to load profile for editing: ${e.message}`);
            closeProfileModal();
        }
    }

    function parseCommaSeparated(val) {
        return val.split(",").map(s => s.trim()).filter(Boolean);
    }

    async function saveProfile() {
        pfSaveBtn.disabled = true;
        pfSaveBtn.textContent = "Saving…";

        try {
            // If a new resume file was attached, upload it first
            let resumeProfileId = null;
            if (pfResumeInput.files.length > 0) {
                const formData = new FormData();
                formData.append("file", pfResumeInput.files[0]);
                const uploadRes = await fetch(`${API_BASE}/api/upload-resume`, {
                    method: "POST",
                    body: formData,
                });
                if (!uploadRes.ok) throw new Error("Resume upload failed");
                const uploadData = await uploadRes.json();
                resumeProfileId = uploadData.profile_id;
            }

            const body = {
                name:                document.getElementById("pf-name").value.trim(),
                email:               document.getElementById("pf-email").value.trim() || null,
                desired_titles:      parseCommaSeparated(document.getElementById("pf-desired").value),
                preferred_locations: parseCommaSeparated(document.getElementById("pf-locations").value),
                remote_preference:   document.getElementById("pf-remote").value || null,
                min_salary:          parseInt(document.getElementById("pf-salary").value, 10) || null,
                industries:          parseCommaSeparated(document.getElementById("pf-industries").value),
            };

            // Include profile ID if editing
            const pfId = document.getElementById("pf-id").value;
            if (pfId) body.id = pfId;
            // If resume was uploaded in this save, use that profile's ID
            if (resumeProfileId && !pfId) body.id = resumeProfileId;

            const res = await fetch(`${API_BASE}/api/profiles/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${res.status}`);
            }

            const saved = await res.json();
            closeProfileModal();
            await loadProfiles();

            // Update badge
            resumeBadge.hidden = false;
            resumeName.textContent = saved.name;

            addMessage("assistant", `Profile **${saved.name}** saved successfully! ${saved.skills_count} skills loaded.`);
        } catch (e) {
            console.error("Save profile error:", e);
            addMessage("error", `Failed to save profile: ${e.message}`);
        } finally {
            pfSaveBtn.disabled = false;
            pfSaveBtn.textContent = "Save Profile";
        }
    }

    // --- Modal event listeners ---
    profileEditBtn.addEventListener("click", () => {
        const selectedId = profileSelect.value;
        if (!selectedId) {
            addMessage("error", "Select a profile first, or click **+ New** to create one.");
            return;
        }
        openProfileModal(selectedId);
    });

    profileNewBtn.addEventListener("click", () => {
        openProfileModal(null);
    });

    modalCloseBtn.addEventListener("click", closeProfileModal);
    modalCancelBtn.addEventListener("click", closeProfileModal);

    // Overlay click to close
    profileModal.addEventListener("click", (e) => {
        if (e.target === profileModal) closeProfileModal();
    });

    // Escape key to close
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !profileModal.hidden) closeProfileModal();
    });

    // Resume file pick
    pfResumeBtn.addEventListener("click", () => pfResumeInput.click());
    pfResumeInput.addEventListener("change", () => {
        if (pfResumeInput.files.length > 0) {
            pfResumeName.textContent = pfResumeInput.files[0].name;
            pfResumeName.classList.add("has-file");
        } else {
            pfResumeName.textContent = "No file chosen";
            pfResumeName.classList.remove("has-file");
        }
    });

    // Handle form submit (covers both Save button click and Enter key)
    profileForm.addEventListener("submit", (e) => {
        e.preventDefault();
        saveProfile();
    });

    // Focus input on page load
    messageInput.focus();
})();
