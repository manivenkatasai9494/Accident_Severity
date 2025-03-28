class ChatWidget {
    constructor() {
        this.userId = 'user_' + Math.random().toString(36).substr(2, 9);
        this.messageTimeout = 30000; // 30 seconds timeout
        this.init();
    }

    init() {
        // Create chat widget HTML
        const chatWidget = document.createElement('div');
        chatWidget.className = 'chat-widget';
        chatWidget.innerHTML = `
            <button class="chat-button" id="chat-toggle">
                <i class="fas fa-comments"></i>
            </button>
            <div class="chat-window" id="chat-window">
                <div class="chat-header">
                    <h3>Emergency Assistant</h3>
                    <button class="close-chat" id="close-chat">&times;</button>
                </div>
                <div class="chat-messages" id="chat-messages"></div>
                <div class="chat-input">
                    <input type="text" id="message-input" placeholder="Type your message...">
                    <button id="send-message">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(chatWidget);

        // Add Font Awesome for icons
        const fontAwesome = document.createElement('link');
        fontAwesome.rel = 'stylesheet';
        fontAwesome.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css';
        document.head.appendChild(fontAwesome);

        // Initialize event listeners
        this.initializeEventListeners();
        
        // Add welcome message
        this.addMessage('Hello! I\'m your emergency response assistant. How can I help you today?', false);
    }

    initializeEventListeners() {
        const chatToggle = document.getElementById('chat-toggle');
        const closeChat = document.getElementById('close-chat');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-message');
        const chatWindow = document.getElementById('chat-window');

        chatToggle.addEventListener('click', () => {
            chatWindow.classList.add('active');
        });

        closeChat.addEventListener('click', () => {
            chatWindow.classList.remove('active');
        });

        sendButton.addEventListener('click', () => this.sendMessage());
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    addMessage(content, isUser) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        messageDiv.textContent = content;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    setLoading(isLoading) {
        const sendButton = document.getElementById('send-message');
        const messageInput = document.getElementById('message-input');
        
        if (isLoading) {
            sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            sendButton.disabled = true;
            messageInput.disabled = true;
        } else {
            sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
            sendButton.disabled = false;
            messageInput.disabled = false;
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('message-input');
        const message = messageInput.value.trim();
        
        if (!message) return;

        // Add user message to chat
        this.addMessage(message, true);
        messageInput.value = '';
        this.setLoading(true);

        // Set timeout for the request
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Request timeout')), this.messageTimeout);
        });

        try {
            const responsePromise = fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    message: message
                })
            });

            const response = await Promise.race([responsePromise, timeoutPromise]);
            const data = await response.json();
            
            if (data.success) {
                this.addMessage(data.response, false);
            } else {
                this.addMessage(data.error || 'Sorry, I encountered an error. Please try again.', false);
            }
        } catch (error) {
            console.error('Chat Error:', error);
            if (error.message === 'Request timeout') {
                this.addMessage('Response taking too long. Please try again.', false);
            } else {
                this.addMessage('Sorry, I encountered an error. Please try again.', false);
            }
        } finally {
            this.setLoading(false);
        }
    }
}

// Initialize chat widget when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatWidget();
}); 