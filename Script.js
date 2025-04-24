// DOM Elements
const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");

// Constants
const API_CONFIG = {
  GROQ: {
    KEY: "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",
    URL: "https://api.groq.com/openai/v1/chat/completions",
    MODEL: "llama3-70b-8192"
  },
  CHROMA: {
    URL: "http://localhost:8000/api/query"
  }
};

// State
const state = {
  userMessage: null,
  inputInitHeight: chatInput.scrollHeight,
  memory: [],
  isGeneratingResponse: false
};

// Memory Management
const MemoryManager = {
  add(role, message) {
    state.memory.push({ role, message });
  },
  
  summarize(maxTokens = 1500) {
    const memoryString = state.memory
      .map(entry => `${entry.role}: ${entry.message}`)
      .join("\n");
    
    return memoryString.length > maxTokens
      ? memoryString.slice(-maxTokens)
      : memoryString;
  }
};

// UI Components
const ChatUI = {
  createMessage(message, className) {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    chatLi.innerHTML = className === "outgoing"
      ? `<p>${message}</p>`
      : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    return chatLi;
  },

  typeText(element, text, delay = 10) {
    let index = 0;
    element.textContent = "";
    
    const type = () => {
      if (index < text.length) {
        element.textContent += text.charAt(index);
        index++;
        setTimeout(type, delay);
      }
    };
    
    type();
  },

  scrollToBottom() {
    chatbox.scrollTo(0, chatbox.scrollHeight);
  }
};

// API Services
const APIService = {
  async queryChromaDB(query, memoryContext) {
    try {
      const response = await fetch(API_CONFIG.CHROMA.URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, memory: memoryContext })
      });

      if (!response.ok) throw new Error('Failed to query ChromaDB.');
      
      const data = await response.json();
      return data.documents;
    } catch (error) {
      console.error('ChromaDB Error:', error);
      return [];
    }
  },

  async generateGroqResponse(prompt) {
    const response = await fetch(API_CONFIG.GROQ.URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_CONFIG.GROQ.KEY}`
      },
      body: JSON.stringify({
        model: API_CONFIG.GROQ.MODEL,
        messages: [{ role: "user", content: prompt }],
        temperature: 0.5,
        max_tokens: 1000
      })
    });

    if (!response.ok) throw new Error('Error generating response from Groq.');
    
    const data = await response.json();
    return data.choices[0].message.content;
  }
};

// Core Chat Logic
const ChatManager = {
  async generateResponse(chatElement) {
    if (state.isGeneratingResponse) return;
    
    state.isGeneratingResponse = true;
    const messageElement = chatElement.querySelector("p");
    
    try {
      const memoryContext = MemoryManager.summarize();
      const relevantContext = await APIService.queryChromaDB(state.userMessage, memoryContext);
      const contextString = relevantContext.join("\n\n");

      const prompt = `
      You are an official virtual assistant for HEC Pakistan. Your purpose is to provide accurate and helpful information about programs, policies, and services.

      Core Responsibilities:
      - Provide accurate, clear and concise information from the given context
      - Maintain a professional yet friendly tone
      - Give direct, actionable answers without unnecessary preambles
      - Help users understand and navigate processes efficiently

      Response Guidelines:
      1. Base responses strictly on the provided context
      2. If information is incomplete:
         - Share what you know confidently
         - Clearly identify any missing details
         - Provide alternative solutions or next steps
      3. Keep responses brief and to the point
      4. Use simple, clear language
      5. Never make assumptions or provide uncertain information

      If a query is outside your knowledge domain, respond with:
      "I apologize, but I don't have enough information to accurately answer your question about [topic]. Let me know if you have questions about other HEC services I can help with."
        Conversation History:
        ${memoryContext}

        Context:
        ${contextString}

        Question: ${state.userMessage}
        Answer:
      `;

      const botResponse = await APIService.generateGroqResponse(prompt);
      
      ChatUI.typeText(messageElement, botResponse);
      MemoryManager.add("assistant", botResponse);
      
    } catch (error) {
      messageElement.textContent = error.message || "Error generating response.";
    } finally {
      state.isGeneratingResponse = false;
      ChatUI.scrollToBottom();
    }
  },

  handleChat() {
    state.userMessage = chatInput.value.trim();
    if (!state.userMessage) return;

    MemoryManager.add("user", state.userMessage);

    chatInput.value = "";
    chatInput.style.height = `${state.inputInitHeight}px`;

    chatbox.appendChild(ChatUI.createMessage(state.userMessage, "outgoing"));
    ChatUI.scrollToBottom();

    setTimeout(() => {
      const incomingChatLi = ChatUI.createMessage("Thinking...", "incoming");
      chatbox.appendChild(incomingChatLi);
      ChatUI.scrollToBottom();
      this.generateResponse(incomingChatLi);
    }, 600);
  }
};

// Event Listeners
chatInput.addEventListener("input", () => {
  chatInput.style.height = `${state.inputInitHeight}px`;
  chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ChatManager.handleChat();
  }
});

sendChatBtn.addEventListener("click", () => ChatManager.handleChat());
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));
