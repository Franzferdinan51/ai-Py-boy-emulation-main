import React, { useState, useCallback, useEffect, useRef } from 'react';
import { MessageCircle, Send, X, Bot, User, Loader2, Sparkles } from 'lucide-react';

interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  provider?: string | null;
  timestamp: string;
}

interface FloatingChatProps {
  backendUrl: string;
  isRomLoaded: boolean;
  provider?: string;
  model?: string;
  apiKey?: string;
  apiEndpoint?: string;
  goal?: string;
  disabled?: boolean;
}

const FloatingChat: React.FC<FloatingChatProps> = ({
  backendUrl,
  isRomLoaded,
  provider = 'bailian',
  model,
  apiKey,
  apiEndpoint,
  goal = '',
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Focus input when panel opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Load recent messages from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem('pyboy_floating_chat_history');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setMessages(parsed.slice(-20)); // Keep last 20 messages
        }
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  // Save messages to localStorage
  useEffect(() => {
    if (messages.length > 0) {
      try {
        localStorage.setItem('pyboy_floating_chat_history', JSON.stringify(messages.slice(-50)));
      } catch {
        // Ignore storage errors
      }
    }
  }, [messages]);

  const sendMessage = useCallback(async () => {
    const trimmedInput = (input || '').trim();
    if (!trimmedInput || isLoading || !isRomLoaded) {
      return;
    }

    const userMessage: ChatMessage = {
      id: Date.now(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: trimmedInput,
          api_name: provider,
          model: model || undefined,
          api_key: apiKey || undefined,
          api_endpoint: apiEndpoint || undefined,
          goal: goal || undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: ChatMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response || data.message || 'No response received',
        provider: data.provider_used || provider,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);

      // Add error as assistant message for visibility
      const errorMessageObj: ChatMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Error: ${errorMessage}`,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessageObj]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, isRomLoaded, backendUrl, provider, model, apiKey, apiEndpoint, goal]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        void sendMessage();
      }
    },
    [sendMessage]
  );

  const clearHistory = useCallback(() => {
    setMessages([]);
    localStorage.removeItem('pyboy_floating_chat_history');
  }, []);

  // Floating button (collapsed state)
  if (!isOpen) {
    return (
      <button
        className="floating-chat-button"
        onClick={() => setIsOpen(true)}
        disabled={disabled}
        title="Open agent chat"
        aria-label="Open floating chat"
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          width: '56px',
          height: '56px',
          borderRadius: '50%',
          border: 'none',
          background: 'linear-gradient(135deg, var(--berry-strong) 0%, var(--berry) 100%)',
          color: 'var(--text-primary)',
          cursor: disabled ? 'not-allowed' : 'pointer',
          boxShadow: '0 8px 32px rgba(125, 61, 99, 0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          transition: 'transform 0.2s ease, box-shadow 0.2s ease',
          opacity: disabled ? 0.5 : 1,
        }}
        onMouseEnter={(e) => {
          if (!disabled) {
            e.currentTarget.style.transform = 'scale(1.1)';
            e.currentTarget.style.boxShadow = '0 12px 40px rgba(125, 61, 99, 0.5)';
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
          e.currentTarget.style.boxShadow = '0 8px 32px rgba(125, 61, 99, 0.4)';
        }}
      >
        <MessageCircle size={24} />
        {messages.length > 0 && (
          <span
            style={{
              position: 'absolute',
              top: '-4px',
              right: '-4px',
              background: 'var(--olive-strong)',
              color: 'var(--text-ink)',
              fontSize: '11px',
              fontWeight: 700,
              minWidth: '20px',
              height: '20px',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {Math.min(messages.length, 99)}
          </span>
        )}
      </button>
    );
  }

  // Expanded chat panel
  return (
    <div
      className="floating-chat-panel"
      style={{
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        width: '380px',
        maxHeight: '520px',
        borderRadius: 'var(--radius-card)',
        background: 'var(--panel)',
        border: '1px solid var(--panel-border)',
        boxShadow: 'var(--panel-shadow)',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 9999,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px 18px',
          borderBottom: '1px solid var(--panel-border-soft)',
          background: 'var(--panel-soft)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, var(--berry-strong) 0%, var(--berry) 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Sparkles size={16} color="var(--text-primary)" />
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: '15px' }}>Agent Chat</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {isRomLoaded ? 'Ready' : 'No ROM loaded'}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {messages.length > 0 && (
            <button
              onClick={clearHistory}
              title="Clear history"
              style={{
                padding: '6px',
                borderRadius: '6px',
                background: 'transparent',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: '12px',
              }}
            >
              Clear
            </button>
          )}
          <button
            onClick={() => setIsOpen(false)}
            title="Close"
            style={{
              padding: '6px',
              borderRadius: '6px',
              background: 'transparent',
              color: 'var(--text-muted)',
              cursor: 'pointer',
            }}
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '12px 16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
          maxHeight: '320px',
          minHeight: '120px',
        }}
      >
        {messages.length === 0 ? (
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-muted)',
              fontSize: '13px',
              textAlign: 'center',
              padding: '20px',
            }}
          >
            <Bot size={32} style={{ marginBottom: '10px', opacity: 0.5 }} />
            <div>Send instructions to the agent</div>
            <div style={{ fontSize: '11px', marginTop: '4px', opacity: 0.7 }}>
              Ask about game state, request actions, or set goals
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                gap: '4px',
              }}
            >
              <div
                style={{
                  maxWidth: '85%',
                  padding: '10px 14px',
                  borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                  background:
                    msg.role === 'user'
                      ? 'linear-gradient(135deg, var(--berry-strong) 0%, var(--berry) 100%)'
                      : 'var(--panel-soft)',
                  color: msg.role === 'user' ? 'var(--text-primary)' : 'var(--text-secondary)',
                  fontSize: '13px',
                  lineHeight: 1.5,
                  wordBreak: 'break-word',
                }}
              >
                {msg.content}
              </div>
              <div
                style={{
                  fontSize: '10px',
                  color: 'var(--text-muted)',
                  padding: '0 4px',
                }}
              >
                {msg.role === 'user' ? 'You' : msg.provider || 'Assistant'}
                {' • '}
                {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div
        style={{
          padding: '12px 16px 16px',
          borderTop: '1px solid var(--panel-border-soft)',
          background: 'var(--panel-soft)',
        }}
      >
        {error && (
          <div
            style={{
              fontSize: '11px',
              color: 'var(--danger)',
              marginBottom: '8px',
              padding: '6px 10px',
              background: 'rgba(255, 140, 125, 0.1)',
              borderRadius: '6px',
            }}
          >
            {error}
          </div>
        )}
        <div
          style={{
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-end',
          }}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isRomLoaded ? 'Type a message...' : 'Load a ROM first'}
            disabled={isLoading || !isRomLoaded}
            rows={1}
            style={{
              flex: 1,
              padding: '12px 14px',
              borderRadius: '12px',
              border: '1px solid var(--panel-border)',
              background: 'var(--panel)',
              color: 'var(--text-primary)',
              fontSize: '13px',
              resize: 'none',
              minHeight: '44px',
              maxHeight: '100px',
              outline: 'none',
              transition: 'border-color 0.2s ease',
            }}
            onFocus={(e) => {
              e.target.style.borderColor = 'var(--berry-strong)';
            }}
            onBlur={(e) => {
              e.target.style.borderColor = 'var(--panel-border)';
            }}
          />
          <button
            onClick={() => void sendMessage()}
            disabled={isLoading || !input.trim() || !isRomLoaded}
            title="Send message"
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '12px',
              border: 'none',
              background:
                isLoading || !input.trim() || !isRomLoaded
                  ? 'var(--panel-border)'
                  : 'linear-gradient(135deg, var(--berry-strong) 0%, var(--berry) 100%)',
              color: 'var(--text-primary)',
              cursor:
                isLoading || !input.trim() || !isRomLoaded ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              opacity: isLoading || !input.trim() || !isRomLoaded ? 0.5 : 1,
              transition: 'opacity 0.2s ease, transform 0.1s ease',
            }}
          >
            {isLoading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
          </button>
        </div>
        <div
          style={{
            fontSize: '10px',
            color: 'var(--text-muted)',
            marginTop: '8px',
            textAlign: 'center',
          }}
        >
          Press Enter to send • Shift+Enter for new line
        </div>
      </div>

      {/* Spinner animation */}
      <style>
        {`
          .spin {
            animation: spin 1s linear infinite;
          }
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default FloatingChat;