const transcriptEl = document.querySelector("#transcript");
const teamFeedEl = document.querySelector("#teamFeed");
const callStatus = document.querySelector("#callStatus");
const modelLabel = document.querySelector("#modelLabel");
const pageDate = document.querySelector("#pageDate");
const micButton = document.querySelector("#micButton");
const cameraButton = document.querySelector("#cameraButton");
const cameraStage = document.querySelector("#cameraStage");
const cameraPreview = document.querySelector("#cameraPreview");
const frameCanvas = document.querySelector("#frameCanvas");
const newIntakeButton = document.querySelector("#newIntakeButton");
const textForm = document.querySelector("#textForm");
const textInput = document.querySelector("#textInput");
const reportDialog = document.querySelector("#reportDialog");
const reportFrame = document.querySelector("#reportFrame");
const localeForm = document.querySelector("#localeForm");
const countryInput = document.querySelector("#countryInput");
const currencyInput = document.querySelector("#currencyInput");

// Typing suggestions only. The server derives the currency from the country (live_demo/locales.py).
const COUNTRY_SUGGESTIONS = [
  "Australia", "Austria", "Belgium", "Brazil", "Canada", "China", "Denmark", "France", "Germany", "India",
  "Ireland", "Italy", "Japan", "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", "Portugal",
  "Singapore", "South Africa", "Spain", "Sweden", "Switzerland", "United Arab Emirates", "United Kingdom", "United States",
];

const DEFAULT_API_ORIGIN = "http://127.0.0.1:4177";
const API_ORIGIN = window.location.protocol === "file:" ? DEFAULT_API_ORIGIN : window.location.origin;
const WS_ORIGIN = API_ORIGIN.replace(/^http/, "ws");
const FRAME_INTERVAL_MS = 1000;
const FRAME_WIDTH = 512;

if (window.location.protocol === "file:") {
  window.location.replace(`${API_ORIGIN}/index.html`);
}

let liveSocket = null;
let connectionPromise = null;
let generation = 0;
let micPending = false;
let cameraPending = false;
let resetting = false;
const playbackSources = new Set();
let audioContext = null;
let inputProcessor = null;
let inputSource = null;
let audioStream = null;
let cameraStream = null;
let frameTimer = null;
let isRecording = false;
let nextPlaybackTime = 0;
let sessionId = null;
let state = null;

const teamLabels = {
  set_country: "Country",
  log_item: "Item log",
  analyse_books: "Books agent",
  get_books: "Books agent",
  exclude_book: "Books agent",
  set_book_copies: "Books agent",
  confirm_book: "Books agent",
  room_step: "Room",
  measure_room: "Room",
  vision: "Object detector",
};

const emptyState = {
  transcript: [],
  tool_activity: [],
  locale: {},
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function setStatus(text, tone = "") {
  callStatus.textContent = text;
  callStatus.className = `pill ${tone}`.trim();
}

function setState(nextState) {
  const previous = state || emptyState;
  state = {
    ...emptyState,
    ...nextState,
    tool_activity: nextState.tool_activity ?? previous.tool_activity,
  };
  syncLocaleInputs();
  render();
}

function syncLocaleInputs() {
  const locale = state.locale || {};
  if (document.activeElement !== countryInput) countryInput.value = locale.country || "";
  if (document.activeElement !== currencyInput) currencyInput.value = locale.currency || "";
}

async function saveLocale() {
  const country = countryInput.value.trim();
  const current = state?.locale || {};
  if (!sessionId || !country || country === (current.country || "")) return;
  if (liveSocket?.readyState === WebSocket.OPEN) {
    liveSocket.send(JSON.stringify({ type: "locale", country }));
    return;
  }
  try {
    const payload = await api(`/api/sessions/${sessionId}/locale`, { method: "PUT", body: JSON.stringify({ country }) });
    state = { ...state, locale: payload.locale };
    syncLocaleInputs();
  } catch (error) {
    appendSystem(error.message);
  }
}

function render() {
  renderTranscript();
  renderTeam();
}

function renderTranscript() {
  transcriptEl.innerHTML = (state.transcript || [])
    .map((turn) => {
      const cls = turn.speaker === "Agent" ? "agent" : turn.speaker === "System" ? "system" : "claimant";
      const who = turn.speaker === "Agent" ? "AI" : turn.speaker === "System" ? "!" : "You";
      return `<article class="turn ${cls} ${turn.streaming ? "streaming" : ""}"><span class="who">${who}</span><p>${escapeHtml(turn.text)}</p></article>`;
    })
    .join("");
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function renderTeam() {
  const activity = [...(state.tool_activity || [])].slice(-6).reverse();
  teamFeedEl.innerHTML = activity.length
    ? activity
        .map((item) => {
          const phase = item.phase || "running";
          const headline = item.headline || "";
          const meta = phase === "running" ? "working" : item.scheduling === "INTERRUPT" ? "interrupted the agent" : item.duration_ms != null ? `${item.duration_ms} ms` : "";
          return `<li><span class="team-dot ${phase}"></span><span><span class="team-name">${escapeHtml(teamLabels[item.name] || item.name)}</span> · ${escapeHtml(headline)}</span><span class="team-meta ${item.scheduling === "INTERRUPT" ? "interrupt" : ""}">${escapeHtml(meta)}</span></li>`;
        })
        .join("")
    : `<li class="empty">The object detector, item log and books agent show up here as they work.</li>`;
}

function appendSystem(text) {
  setState({ ...state, transcript: [...(state.transcript || []), { id: crypto.randomUUID(), speaker: "System", text }] });
}

function mergeTranscript(authoritative, local) {
  const byId = new Map((authoritative || []).map(turn => [turn.id, turn]));
  for (const turn of local || []) {
    if (!byId.has(turn.id)) byId.set(turn.id, turn);
  }
  return [...byId.values()];
}

function applyServerState(nextState) {
  if (nextState.session_id !== sessionId) return;
  setState({
    ...nextState,
    transcript: mergeTranscript(nextState.transcript, state?.transcript),
    tool_activity: mergeToolActivity(nextState.tool_activity, state?.tool_activity),
  });
}

function mergeToolActivity(authoritative, local) {
  const finished = new Set(["done", "error", "cancelled"]);
  const byId = new Map();
  for (const item of authoritative || []) byId.set(item.id, item);
  for (const item of local || []) {
    const current = byId.get(item.id);
    if (!current || (finished.has(item.phase) && !finished.has(current.phase))) byId.set(item.id, item);
  }
  return [...byId.values()];
}

function applyToolEvent(message) {
  const { type, ...entry } = message;
  const others = (state.tool_activity || []).filter((item) => item.id !== entry.id);
  setState({ ...state, tool_activity: [...others, entry] });
}

function upsertStreamingTurn(speaker, text, final = false, id = crypto.randomUUID()) {
  if (!String(text || "").trim()) return;
  const transcript = [...(state.transcript || [])];
  const index = transcript.findIndex(turn => turn.id === id);
  const turn = { id, speaker, text, streaming: !final };
  if (index >= 0) transcript[index] = turn;
  else transcript.push(turn);
  setState({ ...state, transcript });
}

async function api(path, options = {}) {
  const response = await fetch(`${API_ORIGIN}${path}`, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed with status ${response.status}`);
  return payload;
}

function stopPlayback() {
  for (const source of playbackSources) { try { source.stop(); } catch (_) {} }
  playbackSources.clear();
  nextPlaybackTime = audioContext?.currentTime || 0;
}

function clearBusy() {
  state.tool_activity = (state.tool_activity || []).map(item => item.phase === "running" ? { ...item, phase: "cancelled", headline: "Connection ended" } : item);
  render();
}

function disconnectLive() {
  generation += 1;
  const socket = liveSocket;
  liveSocket = null;
  connectionPromise = null;
  stopLiveVoice(false);
  stopCamera();
  stopPlayback();
  if (socket) socket.close();
  if (state) clearBusy();
}

async function createSession(resume = false) {
  if (resetting) return;
  resetting = true;
  newIntakeButton.disabled = true;
  const previous = sessionId || sessionStorage.getItem("intakeSession");
  disconnectLive();
  const epoch = generation;
  setStatus("Connecting", "neutral");
  sessionId = null;
  setState(emptyState);
  pageDate.textContent = `Inventory report · ${new Date().toLocaleDateString([], { month: "short", day: "numeric" })}`;
  try {
    let payload;
    if (resume && previous) {
      try { payload = await api(`/api/sessions/${previous}`); } catch (_) { sessionStorage.removeItem("intakeSession"); }
    } else if (previous) {
      await api(`/api/sessions/${previous}`, { method: "DELETE" }).catch(() => {});
    }
    payload = payload || await api("/api/sessions", { method: "POST" });
    if (epoch !== generation) return;
    sessionId = payload.session_id;
    sessionStorage.setItem("intakeSession", sessionId);
    setState(payload.state);
    const health = await api("/api/health");
    modelLabel.textContent = health.live_model;
    setStatus(payload.has_api_key ? "Ready" : "API key required", payload.has_api_key ? "" : "danger");
    textInput.focus();
  } catch (error) {
    setStatus("Backend unavailable", "danger");
    appendSystem(error.message);
  } finally {
    resetting = false;
    newIntakeButton.disabled = false;
  }
}

function connectLive() {
  if (connectionPromise) return connectionPromise;
  if (!sessionId || resetting) return Promise.reject(new Error("Wait for the intake to be ready."));
  const epoch = generation;
  const socket = new WebSocket(`${WS_ORIGIN}/ws/live?session_id=${encodeURIComponent(sessionId)}`);
  liveSocket = socket;
  connectionPromise = new Promise((resolve, reject) => {
    let ready = false;
    const timeout = setTimeout(() => { reject(new Error("Live connection timed out.")); socket.close(); }, 20000);
    const current = () => epoch === generation && liveSocket === socket;
    socket.onmessage = (event) => {
      if (!current()) return;
      const message = JSON.parse(event.data);
      if (message.session_id !== sessionId) return;
      if (message.type === "ready") {
        ready = true;
        clearTimeout(timeout);
        setStatus("Live", "");
        resolve();
      } else if (message.type === "session") {
        modelLabel.textContent = message.model;
      } else if (message.type === "transcript") {
        upsertStreamingTurn(message.speaker, message.text, message.final, message.id);
      } else if (message.type === "tool") {
        applyToolEvent(message);
      } else if (message.type === "audio") {
        playPcm24(message.data);
      } else if (message.type === "state") {
        applyServerState(message.state);
      } else if (message.type === "report" || message.type === "report_live") {
        showLiveReport();
      } else if (message.type === "interrupted") {
        stopPlayback();
      } else if (message.type === "error") {
        appendSystem(message.message);
        if (!ready) { reject(new Error(message.message)); socket.close(); }
      }
    };
    socket.onerror = () => { clearTimeout(timeout); reject(new Error("Live connection failed.")); socket.close(); };
    socket.onclose = () => {
      clearTimeout(timeout);
      if (!ready) reject(new Error("Live connection ended before it was ready."));
      if (!current()) return;
      liveSocket = null;
      connectionPromise = null;
      stopLiveVoice(false);
      stopCamera();
      stopPlayback();
      clearBusy();
      setStatus("Disconnected — reconnect to continue", "warning");
    };
  });
  return connectionPromise;
}

async function unlockAudio() {
  if (window.AudioContext) {
    audioContext = audioContext || new AudioContext();
    await audioContext.resume();
  }
}

async function sendClaimantTurn(text) {
  const epoch = generation;
  try {
    stopPlayback();
    await unlockAudio();
    await connectLive();
    if (epoch !== generation) return;
    const id = crypto.randomUUID();
    liveSocket.send(JSON.stringify({ type: "text", text, id }));
    upsertStreamingTurn("Claimant", text, true, id);
  } catch (error) {
    if (epoch !== generation) return;
    textInput.value = text;
    appendSystem(`${error.message} Your message is back in the input; send it again to retry.`);
  }
}

async function startLiveVoice() {
  if (micPending || isRecording) return;
  const epoch = generation;
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
    appendSystem("This browser cannot capture microphone audio. Type instead.");
    return;
  }
  try {
    micPending = true;
    await unlockAudio();
    await connectLive();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    if (epoch !== generation || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) { stream.getTracks().forEach(track => track.stop()); return; }
    audioStream = stream;
    audioContext = audioContext || new AudioContext();
    await audioContext.resume();
    inputSource = audioContext.createMediaStreamSource(audioStream);
    inputProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    inputProcessor.onaudioprocess = (event) => {
      event.outputBuffer.getChannelData(0).fill(0);
      if (!liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
      const pcm16 = resampleToPcm16(event.inputBuffer.getChannelData(0), audioContext.sampleRate, 16000);
      liveSocket.send(JSON.stringify({ type: "audio", data: arrayBufferToBase64(pcm16.buffer) }));
    };
    inputSource.connect(inputProcessor);
    inputProcessor.connect(audioContext.destination);
    isRecording = true;
    micButton.classList.add("active");
    micButton.querySelector(".round-label").textContent = "Listening";
    setStatus("Live, listening", "");
  } catch (error) {
    const denied = error.name === "NotAllowedError" || /denied|permission/i.test(error.message);
    appendSystem(denied ? "Microphone access was denied. Allow it for this site or type instead." : `Voice failed: ${error.message}`);
    stopLiveVoice(false);
  } finally { micPending = false; }
}

function stopLiveVoice(closeSocket = true) {
  isRecording = false;
  micButton.classList.remove("active");
  micButton.querySelector(".round-label").textContent = "Talk";
  inputProcessor?.disconnect();
  inputSource?.disconnect();
  audioStream?.getTracks().forEach((track) => track.stop());
  inputProcessor = null;
  inputSource = null;
  audioStream = null;
  if (closeSocket) {
    disconnectLive();
    setStatus("Call ended — intake saved for reconnect", "neutral");
  }
}

async function startCamera() {
  if (cameraPending || cameraStream) return;
  const epoch = generation;
  if (!navigator.mediaDevices?.getUserMedia) {
    appendSystem("This browser cannot access a camera.");
    return;
  }
  try {
    cameraPending = true;
    await unlockAudio();
    await connectLive();
    // High resolution so spines stay readable; the agent's own stream is downscaled to FRAME_WIDTH.
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "environment" } });
    if (epoch !== generation || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) { stream.getTracks().forEach(track => track.stop()); return; }
    cameraStream = stream;
    stream.getVideoTracks().forEach(track => track.addEventListener("ended", stopCamera, { once: true }));
    liveSocket.send(JSON.stringify({ type: "camera_state", enabled: true }));
    cameraPreview.srcObject = cameraStream;
    cameraStage.hidden = false;
    cameraButton.classList.add("active");
    cameraButton.querySelector(".round-label").textContent = "Stop camera";
    frameTimer = window.setInterval(sendFrame, FRAME_INTERVAL_MS);
    if (!isRecording) setStatus("Camera on, agent can see", "");
  } catch (error) {
    const denied = error.name === "NotAllowedError" || /denied|permission/i.test(error.message);
    appendSystem(denied ? "Camera access was denied. Allow it for this site to show the damage." : `Camera failed: ${error.message}`);
    stopCamera();
  } finally { cameraPending = false; }
}

function stopCamera() {
  const wasOn = Boolean(cameraStream);
  if (frameTimer) window.clearInterval(frameTimer);
  frameTimer = null;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
  cameraStage.hidden = true;
  cameraButton.classList.remove("active");
  cameraButton.querySelector(".round-label").textContent = "Show camera";
  if (wasOn && liveSocket?.readyState === WebSocket.OPEN) {
    liveSocket.send(JSON.stringify({ type: "camera_state", enabled: false }));
    if (!isRecording) setStatus("Camera off, describing the scene", "");
  }
}

function sendFrame() {
  if (!cameraStream || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
  if (!cameraPreview.videoWidth) return;
  const scale = FRAME_WIDTH / cameraPreview.videoWidth;
  frameCanvas.width = FRAME_WIDTH;
  frameCanvas.height = Math.round(cameraPreview.videoHeight * scale);
  const ctx = frameCanvas.getContext("2d");
  ctx.drawImage(cameraPreview, 0, 0, frameCanvas.width, frameCanvas.height);
  const dataUrl = frameCanvas.toDataURL("image/jpeg", 0.6);
  liveSocket.send(JSON.stringify({ type: "video", data: dataUrl.split(",")[1] }));
  uploadHighRes();
}

// Full-resolution frame for live detection and the book agent; the server keeps only spine frames.
let highResInFlight = false;
function uploadHighRes() {
  if (highResInFlight || !sessionId || !cameraPreview.videoWidth) return;
  const canvas = uploadHighRes.canvas || (uploadHighRes.canvas = document.createElement("canvas"));
  canvas.width = cameraPreview.videoWidth;
  canvas.height = cameraPreview.videoHeight;
  canvas.getContext("2d").drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);
  highResInFlight = true;
  const target = sessionId;
  canvas.toBlob(async (blob) => {
    try {
      if (blob && target === sessionId) {
        await fetch(`${API_ORIGIN}/api/sessions/${target}/frames`, { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
      }
    } catch (_) {
      // A dropped frame is fine; the next one follows within a second.
    } finally {
      highResInFlight = false;
    }
  }, "image/jpeg", 0.85);
}

function resampleToPcm16(input, inputRate, outputRate) {
  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(input.length / ratio);
  const pcm = new Int16Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const index = i * ratio;
    const before = Math.floor(index);
    const after = Math.min(before + 1, input.length - 1);
    const weight = index - before;
    const sample = Math.max(-1, Math.min(1, input[before] * (1 - weight) + input[after] * weight));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm;
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function playPcm24(base64) {
  audioContext = audioContext || new AudioContext();
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  const pcm = new Int16Array(bytes.buffer);
  const audioBuffer = audioContext.createBuffer(1, pcm.length, 24000);
  const channel = audioBuffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i += 1) channel[i] = pcm[i] / 32768;
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  const startAt = Math.max(audioContext.currentTime, nextPlaybackTime);
  playbackSources.add(source);
  source.onended = () => playbackSources.delete(source);
  source.start(startAt);
  nextPlaybackTime = startAt + audioBuffer.duration;
}

document.querySelector("#countryOptions").innerHTML = COUNTRY_SUGGESTIONS.map(name => `<option value="${escapeHtml(name)}"></option>`).join("");
countryInput.addEventListener("change", saveLocale);
localeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  saveLocale();
});
micButton.addEventListener("click", () => (isRecording ? stopLiveVoice() : startLiveVoice()));
cameraButton.addEventListener("click", () => (cameraStream ? stopCamera() : startCamera()));
newIntakeButton.addEventListener("click", () => createSession(false));
window.addEventListener("pagehide", disconnectLive);
textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = textInput.value.trim();
  if (!value) return;
  textInput.value = "";
  sendClaimantTurn(value);
});
// The report is rebuilt on every load. While the team is still working (the page says so in its
// report-state meta tag), it reloads every 10 s and keeps the reader's scroll position.
const REPORT_REFRESH_MS = 10000;
let reportTimer = null;
let reportScroll = 0;

reportFrame.addEventListener("load", () => {
  const doc = reportFrame.contentDocument;
  if (!doc) return;
  doc.scrollingElement && (doc.scrollingElement.scrollTop = reportScroll);
  const state = doc.querySelector('meta[name="report-state"]')?.content;
  clearTimeout(reportTimer);
  if (state === "in-progress" && reportDialog.open) {
    reportTimer = setTimeout(() => refreshReport(), REPORT_REFRESH_MS);
  }
});

function refreshReport() {
  if (!sessionId || !reportDialog.open) return;
  reportScroll = reportFrame.contentDocument?.scrollingElement?.scrollTop || 0;
  reportFrame.src = `${API_ORIGIN}/api/sessions/${sessionId}/report?t=${Date.now()}`;
}

// The live report in the centre panel: opened when the team first identifies something, then
// refreshed every 10 s while the call is live or the report says work is still in progress.
const liveReport = document.querySelector("#liveReport");
let liveTimer = null;
let liveScroll = 0;

function liveCallOpen() {
  return liveSocket && liveSocket.readyState === WebSocket.OPEN;
}

function loadLiveReport() {
  if (!sessionId || liveReport.hidden) return;
  liveScroll = liveReport.contentDocument?.scrollingElement?.scrollTop || liveScroll;
  liveReport.src = `${API_ORIGIN}/api/sessions/${sessionId}/report?t=${Date.now()}`;
}

function showLiveReport() {
  if (!sessionId) return;
  document.querySelector(".notebook-wrap").classList.add("has-report");
  const firstShow = liveReport.hidden;
  liveReport.hidden = false;
  if (firstShow) liveScroll = 0;
  loadLiveReport();
}

liveReport.addEventListener("load", () => {
  const doc = liveReport.contentDocument;
  if (!doc) return;
  if (doc.scrollingElement) doc.scrollingElement.scrollTop = liveScroll;
  const state = doc.querySelector('meta[name="report-state"]')?.content;
  clearTimeout(liveTimer);
  if (liveCallOpen() || state === "in-progress") liveTimer = setTimeout(loadLiveReport, REPORT_REFRESH_MS);
});

function openReport(savedPath = "") {
  if (!sessionId) return;
  const url = `${API_ORIGIN}/api/sessions/${sessionId}/report`;
  reportScroll = reportDialog.open ? (reportFrame.contentDocument?.scrollingElement?.scrollTop || 0) : 0;
  reportFrame.src = `${url}?t=${Date.now()}`;
  document.querySelector("#reportNewTab").href = url;
  document.querySelector("#downloadClaimPacket").href = `${API_ORIGIN}/api/sessions/${sessionId}/claim_packet.json`;
  if (savedPath) document.querySelector("#reportSaved").textContent = `Saved on this computer: ${savedPath}`;
  if (!reportDialog.open) reportDialog.showModal();
}
document.querySelector("#openReport").addEventListener("click", () => openReport());
document.querySelector("#reloadReport").addEventListener("click", () => openReport());
document.querySelector("#closeReport").addEventListener("click", () => { clearTimeout(reportTimer); reportDialog.close(); });

createSession(true);
