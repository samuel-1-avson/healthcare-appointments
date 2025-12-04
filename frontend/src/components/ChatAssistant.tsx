import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'react-router-dom';
import {
    Send,
    Loader,
    Bot,
    User,
    Sparkles,
    Copy,
    RefreshCw,
    MessageSquare,
    ArrowRight,
    Check,
    AlertTriangle,
    Activity
} from 'lucide-react';
import { chatWithAssistant } from '../services/api';
import type { ChatMessage } from '../types';

const QuickPrompt = ({ text, onClick }: { text: string, onClick: () => void }) => (
    <button
        onClick={onClick}
        className="text-left p-4 rounded-xl bg-surface border border-surface-hover hover:border-primary/50 hover:bg-surface-hover transition-all group"
    >
        <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-text-primary group-hover:text-primary transition-colors">{text}</p>
            <ArrowRight className="w-4 h-4 text-surface-hover group-hover:text-primary opacity-0 group-hover:opacity-100 transition-all transform -translate-x-2 group-hover:translate-x-0" />
        </div>
    </button>
);

const ChatAssistant = () => {
    const location = useLocation();
    const predictionContext = (location.state as any)?.predictionContext;

    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string>('');
    const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
    const [hasInitializedContext, setHasInitializedContext] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        setSessionId(`session-${Date.now()}`);
    }, []);

    // Auto-send context-aware greeting when prediction context is present
    useEffect(() => {
        if (predictionContext && sessionId && !hasInitializedContext && messages.length === 0) {
            setHasInitializedContext(true);
            const contextMessage = `I just made a prediction with the following results:
- Risk Level: ${predictionContext.risk?.tier || 'Unknown'}
- Probability: ${(predictionContext.probability * 100).toFixed(0)}%
- Recommended Action: ${predictionContext.intervention?.action || 'Standard'}

Can you explain why this patient has this risk level and what I should do next?`;
            handleSendMessage(contextMessage);
        }
    }, [predictionContext, sessionId, hasInitializedContext, messages.length]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSendMessage = async (text: string) => {
        if (!text.trim() || isLoading) return;

        const userMessage: ChatMessage = {
            role: 'user',
            content: text,
            timestamp: new Date().toISOString(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await chatWithAssistant({
                message: text,
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

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        handleSendMessage(input);
    };

    const handleCopy = (content: string, index: number) => {
        navigator.clipboard.writeText(content);
        setCopiedIndex(index);
        setTimeout(() => setCopiedIndex(null), 2000);
    };

    const prompts = [
        "What are the top risk factors for no-shows?",
        "How does the model predict patient attendance?",
        "Explain the impact of lead time on attendance.",
        "Draft a reminder message for a high-risk patient."
    ];

    return (
        <div className="h-[calc(100vh-6rem)] flex flex-col max-w-5xl mx-auto relative">
            {/* Header */}
            <div className="flex-none mb-6 text-center">
                <div className="inline-flex items-center justify-center p-3 bg-gradient-to-tr from-primary/20 to-accent/20 rounded-2xl mb-4 backdrop-blur-sm border border-white/5">
                    <Sparkles className="w-6 h-6 text-primary" />
                </div>
                <h1 className="text-3xl font-bold text-text-primary mb-2 tracking-tight">
                    AI Healthcare Assistant
                </h1>
                <p className="text-text-secondary max-w-lg mx-auto">
                    Your intelligent companion for analyzing patient risk and optimizing schedule efficiency.
                </p>
            </div>

            <div className="flex-1 glass-card rounded-3xl flex flex-col overflow-hidden border border-surface-hover shadow-2xl relative">
                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth custom-scrollbar">
                    {messages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center p-8 text-center space-y-8 opacity-0 animate-fade-in" style={{ animationFillMode: 'forwards' }}>
                            <div className="w-24 h-24 bg-surface rounded-full flex items-center justify-center mb-4 shadow-inner">
                                <Bot className="w-12 h-12 text-primary/50" />
                            </div>
                            <div className="space-y-2 max-w-md">
                                <h3 className="text-xl font-bold text-text-primary">How can I help you today?</h3>
                                <p className="text-text-secondary text-sm">
                                    I can analyze patient data, explain risk factors, or help you draft communication strategies.
                                </p>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
                                {prompts.map((prompt, idx) => (
                                    <QuickPrompt key={idx} text={prompt} onClick={() => handleSendMessage(prompt)} />
                                ))}
                            </div>
                        </div>
                    ) : (
                        <AnimatePresence initial={false}>
                            {messages.map((message, idx) => (
                                <motion.div
                                    key={idx}
                                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    transition={{ duration: 0.3 }}
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

                                    <div className={`flex-1 max-w-[80%] space-y-1 group`}>
                                        <div className={`relative rounded-2xl p-5 shadow-sm ${message.role === 'user'
                                            ? 'bg-primary text-white rounded-tr-none'
                                            : 'bg-surface/80 backdrop-blur-md border border-surface-hover text-text-primary rounded-tl-none'
                                            }`}>
                                            <p className="leading-relaxed whitespace-pre-wrap">{message.content}</p>

                                            {message.role === 'assistant' && (
                                                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                                                    <button
                                                        onClick={() => handleCopy(message.content, idx)}
                                                        className="p-1.5 rounded-lg hover:bg-surface-hover text-text-tertiary hover:text-primary transition-colors"
                                                        title="Copy to clipboard"
                                                    >
                                                        {copiedIndex === idx ? <Check className="w-3 h-3 text-success" /> : <Copy className="w-3 h-3" />}
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                        <p className={`text-xs text-text-secondary ${message.role === 'user' ? 'text-right' : 'text-left'
                                            }`}>
                                            {message.timestamp ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                                        </p>
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    )}

                    {isLoading && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="flex gap-4"
                        >
                            <div className="flex-none w-10 h-10 rounded-full bg-surface border border-surface-hover flex items-center justify-center shadow-lg">
                                <Bot className="w-5 h-5 text-primary" />
                            </div>
                            <div className="bg-surface border border-surface-hover rounded-2xl rounded-tl-none p-4 flex items-center gap-2">
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                        </motion.div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="flex-none p-6 bg-gradient-to-t from-background to-transparent absolute bottom-0 left-0 right-0 z-10">
                    <div className="max-w-4xl mx-auto">
                        <form onSubmit={handleSubmit} className="relative flex items-end gap-3 p-2 bg-surface/80 backdrop-blur-xl border border-surface-hover rounded-3xl shadow-2xl ring-1 ring-white/5">
                            <div className="flex-none p-2">
                                <div className="w-8 h-8 rounded-full bg-surface-hover flex items-center justify-center text-text-secondary">
                                    <MessageSquare className="w-4 h-4" />
                                </div>
                            </div>
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask anything..."
                                disabled={isLoading}
                                className="flex-1 py-3 bg-transparent text-text-primary placeholder-text-secondary focus:outline-none min-h-[48px]"
                            />
                            <button
                                type="submit"
                                disabled={!input.trim() || isLoading}
                                className="flex-none p-3 bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-2xl transition-all shadow-lg shadow-primary/25 flex items-center justify-center group"
                            >
                                {isLoading ? (
                                    <Loader className="w-5 h-5 animate-spin" />
                                ) : (
                                    <Send className="w-5 h-5 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                                )}
                            </button>
                        </form>
                        <p className="text-center text-[10px] text-text-tertiary mt-3">
                            AI can make mistakes. Please verify important medical information.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatAssistant;
