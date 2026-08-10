document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const messagesArea = document.getElementById('messages-area');
    const sendBtn = document.getElementById('send-btn');
    const heroSection = document.getElementById('hero-section');
    
    let isFirstMessage = true;

    // Sidebar Toggle
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('toggle-sidebar-btn');
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
    }

    // Toast Notification System
    function showToast(message) {
        const container = document.getElementById('toast-container');
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerText = message;
        
        container.appendChild(toast);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'fadeOut 0.3s ease-out forwards';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // Interactive Nav Items
    const navItems = document.querySelectorAll('.interactive-nav');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active from all siblings
            document.querySelectorAll('.sidebar-top .nav-item').forEach(nav => {
                nav.classList.remove('active');
            });
            item.classList.add('active');
            
            const title = item.getAttribute('data-title') || 'Feature';
            showToast(`${title} feature coming soon!`);
        });
    });

    // Expose sendSuggestion to global scope so inline onclick can use it
    window.sendSuggestion = (text) => {
        messageInput.value = text;
        chatForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    };

    // Create a message element
    function createMessageElement(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Use marked.js for rich formatting if available, fallback to raw text otherwise
        if (typeof marked !== 'undefined') {
            contentDiv.innerHTML = marked.parse(content);
        } else {
            contentDiv.innerText = content;
        }
        
        messageDiv.appendChild(contentDiv);
        
        return messageDiv;
    }

    // Add typing indicator
    function addTypingIndicator() {
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'message assistant typing-indicator-container';
        indicatorDiv.id = 'typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'dot';
            contentDiv.appendChild(dot);
        }
        
        indicatorDiv.appendChild(contentDiv);
        messagesArea.appendChild(indicatorDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function scrollToBottom() {
        // Find the chat-wrapper to scroll if necessary, or messages-area
        messagesArea.scrollTop = messagesArea.scrollHeight;
        // Also scroll the parent chat-wrapper just in case
        const chatWrapper = document.querySelector('.chat-wrapper');
        if(chatWrapper) {
            chatWrapper.scrollTop = chatWrapper.scrollHeight;
        }
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const messageText = messageInput.value.trim();
        if (!messageText) return;

        // Hide hero section on first message
        if (isFirstMessage) {
            heroSection.classList.add('hidden');
            messagesArea.classList.remove('hidden');
            isFirstMessage = false;
        }

        // 1. Add user message to UI
        const userMessageEl = createMessageElement(messageText, 'user');
        messagesArea.appendChild(userMessageEl);
        
        // 2. Clear input
        messageInput.value = '';
        messageInput.disabled = true;
        sendBtn.disabled = true;
        
        scrollToBottom();

        // 3. Add typing indicator
        addTypingIndicator();

        try {
            // 4. Send request to backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: messageText })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // 5. Remove typing indicator and show assistant message
            removeTypingIndicator();
            const assistantMessageEl = createMessageElement(data.response, 'assistant');
            messagesArea.appendChild(assistantMessageEl);

        } catch (error) {
            console.error('Error fetching response:', error);
            removeTypingIndicator();
            const errorMessageEl = createMessageElement('Sorry, I encountered an error while fetching the data. Please try again.', 'assistant');
            messagesArea.appendChild(errorMessageEl);
        } finally {
            messageInput.disabled = false;
            sendBtn.disabled = false;
            messageInput.focus();
            scrollToBottom();
        }
    });
});
