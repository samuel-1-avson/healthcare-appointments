import { useState, useEffect, useRef } from 'react';
import { Send, Loader, Bot, User } from 'lucide-react';
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
        <div className="max-w-4xl mx-auto">
            <div className="mb-6 text-center">
                <h1 className="text-4xl font-extrabold text-gray-900 dark:text-white mb-4">
                    AI <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">Assistant</span>
                </h1>
                <p className="text-xl text-gray-500 dark:text-gray-400">
                    Get instant answers powered by AI
                </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                <div className="h-96 overflow-y-auto mb-4 space-y-4">
                    {messages.map((message, idx) => (
                        <div key={idx} className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${message.role === 'user' ? 'bg-blue-600' : 'bg-purple-600'
                                }`}>
                                {message.role === 'user' ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-white" />}
                            </div>
                            <div className={`flex-1 max-w-[80%] rounded-lg p-4 ${message.role === 'user' ? 'bg-blue-100 dark:bg-blue-900' : 'bg-gray-100 dark:bg-gray-700'
                                }`}>
                                <p className="text-gray-900 dark:text-white">{message.content}</p>
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
                                <Bot className="w-4 h-4 text-white" />
                            </div>
                            <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                                <Loader className="w-5 h-5 animate-spin" />
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <form onSubmit={handleSubmit} className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask me anything..."
                        disabled={isLoading}
                        className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
                    >
                        {isLoading ? <Loader className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatAssistant;
