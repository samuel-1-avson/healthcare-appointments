import { useState, useEffect, useRef } from 'react';
import { Send, Loader, Bot, User, Sparkles } from 'lucide-react';
import { chatWithAssistant } from '../services/api';
import type { ChatMessage } from '../types';

const ChatAssistant = () => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string>('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        setSessionId(`session-${Date.now()}`);
        setMessages([{
            role: 'assistant',
            content: 'Hello! I am your AI Healthcare Assistant. How can I help you today?',
            timestamp: new Date().toISOString(),
        }]);
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: ChatMessage = {
            role: 'user',
            content: input,
            timestamp: new Date().toISOString(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await chatWithAssistant({
                message: input,
                session_id: sessionId,
            });

            const assistantMessage: ChatMessage = {
                role: 'assistant',
                content: response.response,
                timestamp: new Date().toISOString(),
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            const errorMessage: ChatMessage = {
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date().toISOString(),
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="h-[calc(100vh-6rem)] flex flex-col max-w-5xl mx-auto">
            <div className="flex-none mb-6 text-center">
                <div className="inline-flex items-center justify-center p-3 bg-primary/10 rounded-2xl mb-4">
                    <Sparkles className="w-8 h-8 text-primary" />
                </div>
                <h1 className="text-3xl font-bold text-text-primary mb-2">
                    AI Assistant
                </h1>
                <p className="text-text-secondary">
                    Get instant answers about patient risk factors and predictions
                </p>
            </div>

            <div className="flex-1 glass-card rounded-3xl flex flex-col overflow-hidden border border-surface-hover shadow-2xl">
                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
                    {messages.map((message, idx) => (
                        <div
                            key={idx}
                            className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
                        >
                            <div className={`flex-none w-10 h-10 rounded-full flex items-center justify-center shadow-lg ${message.role === 'user'
                                    ? 'bg-gradient-to-tr from-primary to-accent'
                                    : 'bg-surface border border-surface-hover'
                                }`}>
                                {message.role === 'user'
                                    ? <User className="w-5 h-5 text-white" />
                                    : <Bot className="w-5 h-5 text-primary" />
                                }
                            </div>

                            <div className={`flex-1 max-w-[80%] space-y-1`}>
                                <div className={`rounded-2xl p-4 shadow-sm ${message.role === 'user'
                                        ? 'bg-primary text-white rounded-tr-none'
                                        : 'bg-surface border border-surface-hover text-text-primary rounded-tl-none'
                                    }`}>
                                    <p className="leading-relaxed">{message.content}</p>
                                </div>
                                <p className={`text-xs text-text-secondary ${message.role === 'user' ? 'text-right' : 'text-left'
                                    }`}>
                                    {message.timestamp ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                                </p>
                            </div>
                        </div>
                    ))}

                    {isLoading && (
                        <div className="flex gap-4">
                            <div className="flex-none w-10 h-10 rounded-full bg-surface border border-surface-hover flex items-center justify-center shadow-lg">
                                <Bot className="w-5 h-5 text-primary" />
                            </div>
                            <div className="bg-surface border border-surface-hover rounded-2xl rounded-tl-none p-4 flex items-center gap-2">
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="flex-none p-4 bg-surface/50 border-t border-surface-hover backdrop-blur-md">
                    <form onSubmit={handleSubmit} className="flex gap-3 max-w-4xl mx-auto">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask about risk factors, predictions, or recommendations..."
                            disabled={isLoading}
                            className="flex-1 px-6 py-4 bg-surface border border-surface-hover rounded-2xl text-text-primary placeholder-text-secondary focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all shadow-inner"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="flex-none px-6 py-4 bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-2xl transition-all shadow-lg shadow-primary/25 flex items-center justify-center group"
                        >
                            {isLoading ? (
                                <Loader className="w-6 h-6 animate-spin" />
                            ) : (
                                <Send className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
                            )}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default ChatAssistant;
