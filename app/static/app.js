const topicChips = [
  "Placement year preparation", "Finding a placement", "Placement assessment", "Support during placement"
];
const suggestedPrompts = [
  "I feel anxious about coursework. What support is recommended?",
  "I feel homesick. Who can I contact?",
  "How should I prepare for my placement year?",
  "How is the placement year supported and assessed?"
];
const welcomeMessage = {
  id: "welcome", role: "assistant", category: "UEA Support",
  content: "Hello, How can I help you today."
};

const state = { messages: [welcomeMessage], sending: false, emotionGroups: [] };
const messagesElement = document.querySelector("#messages");
const promptRow = document.querySelector("#prompt-row");
const input = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-button");
const composer = document.querySelector("#composer");
const sidebar = document.querySelector("#sidebar");
const emotionGroupSelect = document.querySelector("#emotion-group");
const emotionReasonSelect = document.querySelector("#emotion-reason");
const emotionSendButton = document.querySelector("#emotion-send");
const emotionStatus = document.querySelector("#emotion-status");

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function makeOption(value, text) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = text;
  return option;
}

function renderMessage(message) {
  const article = makeElement("article", `message ${message.role}`);
  article.dataset.messageId = message.id;
  if (message.role === "user") {
    article.append(makeElement("div", "message-meta", "You"));
  }
  article.append(makeElement("p", null, message.content));
  if (message.safetyWarning) {
    const warning = makeElement("aside", "safety-warning");
    warning.setAttribute("role", "alert");
    warning.append(
      makeElement("strong", null, "Please get urgent human support now."),
      makeElement("p", null, "If you may act now or cannot stay safe, call 999 or go to A&E. You can also call NHS 111 and select the mental health option, or Samaritans free on 116 123.")
    );
    article.append(warning);
  }
  return article;
}

function renderMessages() {
  messagesElement.replaceChildren(...state.messages.map(renderMessage));
  if (state.sending) {
    const typing = makeElement("article", "message assistant typing");
    const dots = makeElement("div", "typing-dots");
    dots.setAttribute("aria-label", "Assistant is typing");
    dots.append(makeElement("span"), makeElement("span"), makeElement("span"));
    typing.append(dots);
    messagesElement.append(typing);
  }
  messagesElement.scrollTop = messagesElement.scrollHeight;
  promptRow.hidden = state.messages.some((message) => message.role === "user");
}

function choosePrompt(prompt) {
  input.value = prompt;
  updateComposer();
  input.focus();
}

function updateComposer() {
  sendButton.disabled = !input.value.trim() || state.sending;
}

function updateEmotionSendButton() {
  emotionSendButton.disabled = !emotionGroupSelect.value || !emotionReasonSelect.value || state.sending;
}

function updateEmotionControls() {
  const group = state.emotionGroups.find((item) => item.group === emotionGroupSelect.value);
  emotionReasonSelect.replaceChildren(makeOption("", group ? "Choose a reason" : "Choose a feeling first"));
  emotionReasonSelect.disabled = !group || state.sending;

  if (group) {
    group.options.forEach((option, index) => {
      emotionReasonSelect.append(makeOption(String(index), option.emotion));
    });
  }

  updateEmotionSendButton();
}

function renderEmotionPicker() {
  emotionGroupSelect.replaceChildren(makeOption("", "Choose a feeling"));
  state.emotionGroups.forEach((group) => {
    emotionGroupSelect.append(makeOption(group.group, group.group));
  });
  emotionGroupSelect.disabled = state.emotionGroups.length === 0;
  updateEmotionControls();
}

async function loadEmotionPicker() {
  try {
    const response = await fetch("/api/emotions");
    const body = await response.json().catch(() => []);
    if (!response.ok) throw new Error(body.detail || "Unable to load feelings.");
    state.emotionGroups = body;
    emotionStatus.textContent = "";
    renderEmotionPicker();
  } catch (error) {
    emotionStatus.textContent = error.message || "Feelings are unavailable.";
    emotionGroupSelect.disabled = true;
    emotionReasonSelect.disabled = true;
    emotionSendButton.disabled = true;
  }
}

function sendEmotionSelection() {
  const group = state.emotionGroups.find((item) => item.group === emotionGroupSelect.value);
  const option = group?.options[Number(emotionReasonSelect.value)];
  if (!option || state.sending) return;
  const reason = option.reason || "this reason";
  input.value = `I am feeling ${option.emotion} because ${reason}. What support is recommended?`;
  updateComposer();
  composer.requestSubmit();
}

async function sendMessage(event) {
  event.preventDefault();
  const userText = input.value.trim();
  if (!userText || state.sending) return;
  const history = state.messages
    .filter((message) => message.id !== "welcome" && !message.isError)
    .slice(-20)
    .map(({ role, content }) => ({ role, content }));
  state.messages.push({ id: `user-${Date.now()}`, role: "user", content: userText });
  state.sending = true;
  input.value = "";
  updateComposer();
  updateEmotionControls();
  renderMessages();
  try {
    const response = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userText, history })
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "Unable to get a response.");
    state.messages.push(body);
  } catch (error) {
    state.messages.push({
      id: `error-${Date.now()}`, role: "assistant", category: "Connection",
      content: error.message || "I could not reach the support service. Please try again in a moment.",
      isError: true
    });
  } finally {
    state.sending = false;
    updateComposer();
    updateEmotionControls();
    renderMessages();
    input.focus();
  }
}

topicChips.forEach((topic) => {
  const button = makeElement("button", "topic-chip", topic);
  button.type = "button";
  button.addEventListener("click", () => choosePrompt(`I need help with ${topic.toLowerCase()}.`));
  document.querySelector("#topic-chips").append(button);
});
suggestedPrompts.forEach((prompt) => {
  const button = makeElement("button", "prompt-chip", prompt);
  button.type = "button";
  button.addEventListener("click", () => choosePrompt(prompt));
  promptRow.append(button);
});
document.querySelectorAll(".nav-pill").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-pill").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    choosePrompt(`I would like ${button.dataset.topic.toLowerCase()} support.`);
  });
});
document.querySelector("#open-menu").addEventListener("click", () => sidebar.classList.add("is-open"));
document.querySelector("#close-menu").addEventListener("click", () => sidebar.classList.remove("is-open"));
emotionGroupSelect.addEventListener("change", updateEmotionControls);
emotionReasonSelect.addEventListener("change", updateEmotionSendButton);
emotionSendButton.addEventListener("click", sendEmotionSelection);
input.addEventListener("input", updateComposer);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});
composer.addEventListener("submit", sendMessage);
renderMessages();
loadEmotionPicker();
