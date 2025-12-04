import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
    LayoutDashboard,
    Activity,
    Upload,
    MessageSquare,
    Menu,
    X,
    Bell,
    ChevronRight,
    Settings,
    Users
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
    return twMerge(clsx(inputs));
}

interface LayoutProps {
    children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
    const location = useLocation();

    const navigation = [
        { name: 'Dashboard', href: '/', icon: LayoutDashboard },
        { name: 'Predict', href: '/predict', icon: Activity },
        { name: 'Batch Upload', href: '/batch', icon: Upload },
        { name: 'Patients', href: '/patients', icon: Users },
        { name: 'Assistant', href: '/chat', icon: MessageSquare },
        { name: 'Settings', href: '/settings', icon: Settings },
    ];

    return (
        <div className="min-h-screen bg-background text-text-primary flex font-sans">
            {/* Sidebar */}
            <aside
                className={cn(
                    "fixed inset-y-0 left-0 z-50 w-64 glass transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0",
                    isSidebarOpen ? "translate-x-0" : "-translate-x-full"
                )}
            >
                <div className="h-full flex flex-col">
                    {/* Logo */}
                    <div className="h-16 flex items-center px-6 border-b border-surface-hover">
                        <Activity className="h-8 w-8 text-primary mr-3" />
                        <span className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                            NoShow<span className="font-light text-text-primary">Predict</span>
                        </span>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 px-4 py-6 space-y-2">
                        {navigation.map((item) => {
                            const isActive = location.pathname === item.href;
                            return (
                                <Link
                                    key={item.name}
                                    to={item.href}
                                    className={cn(
                                        "flex items-center px-4 py-3 rounded-xl transition-all duration-200 group btn-hover",
                                        isActive
                                            ? "nav-item-active shadow-glow"
                                            : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
                                    )}
                                >
                                    <item.icon className={cn(
                                        "h-5 w-5 mr-3 transition-colors",
                                        isActive ? "text-primary" : "text-text-secondary group-hover:text-text-primary"
                                    )} />
                                    <span className="font-medium">{item.name}</span>
                                    {isActive && (
                                        <ChevronRight className="ml-auto h-4 w-4 text-primary opacity-100" />
                                    )}
                                </Link>
                            );
                        })}
                    </nav>

                    {/* User Profile Snippet */}
                    <div className="p-4 border-t border-surface-hover">
                        <div className="flex items-center p-3 rounded-xl bg-surface-hover/50 hover:bg-surface-hover transition-colors cursor-pointer">
                            <div className="h-10 w-10 rounded-full bg-gradient-to-tr from-primary to-accent flex items-center justify-center text-white font-bold shadow-lg">
                                DA
                            </div>
                            <div className="ml-3">
                                <p className="text-sm font-medium text-text-primary">Dr. Admin</p>
                                <p className="text-xs text-text-secondary">Administrator</p>
                            </div>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Header */}
                <header className="h-16 glass sticky top-0 z-40 px-6 flex items-center justify-between">
                    <button
                        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                        className="lg:hidden p-2 rounded-lg text-text-secondary hover:bg-surface-hover"
                    >
                        {isSidebarOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                    </button>

                    <div className="flex items-center ml-auto space-x-4">
                        <div className="relative">
                            <button
                                onClick={() => setIsNotificationsOpen(!isNotificationsOpen)}
                                className="p-2 rounded-full text-text-secondary hover:text-primary hover:bg-primary/10 transition-colors relative btn-hover"
                            >
                                <Bell className="h-5 w-5" />
                                <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-error animate-pulse"></span>
                            </button>

                            {/* Notifications Dropdown */}
                            {isNotificationsOpen && (
                                <div className="absolute right-0 mt-2 w-80 glass-card rounded-xl shadow-2xl overflow-hidden z-50 animate-fade-in">
                                    <div className="p-4 border-b border-surface-hover flex justify-between items-center">
                                        <h3 className="font-bold text-text-primary">Notifications</h3>
                                        <span className="text-xs text-primary cursor-pointer hover:underline">Mark all read</span>
                                    </div>
                                    <div className="max-h-64 overflow-y-auto">
                                        <div className="p-4 border-b border-surface-hover hover:bg-surface-hover/30 transition-colors cursor-pointer">
                                            <div className="flex items-start gap-3">
                                                <div className="p-2 bg-error/10 rounded-lg shrink-0">
                                                    <Activity className="w-4 h-4 text-error" />
                                                </div>
                                                <div>
                                                    <p className="text-sm font-medium text-text-primary">High Risk Patient Detected</p>
                                                    <p className="text-xs text-text-secondary mt-1">Patient #12345 flagged as CRITICAL risk.</p>
                                                    <p className="text-xs text-text-secondary mt-2">2 mins ago</p>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="p-4 hover:bg-surface-hover/30 transition-colors cursor-pointer">
                                            <div className="flex items-start gap-3">
                                                <div className="p-2 bg-primary/10 rounded-lg shrink-0">
                                                    <Upload className="w-4 h-4 text-primary" />
                                                </div>
                                                <div>
                                                    <p className="text-sm font-medium text-text-primary">Batch Upload Complete</p>
                                                    <p className="text-xs text-text-secondary mt-1">Successfully processed 150 records.</p>
                                                    <p className="text-xs text-text-secondary mt-2">1 hour ago</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="p-3 text-center border-t border-surface-hover bg-surface-hover/10">
                                        <Link to="/patients" className="text-sm text-primary hover:text-primary-hover font-medium" onClick={() => setIsNotificationsOpen(false)}>
                                            View All Alerts
                                        </Link>
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="h-8 w-px bg-surface-hover mx-2"></div>
                        <div className="flex items-center space-x-2">
                            <span className="text-sm text-text-secondary hidden sm:inline-block">
                                {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
                            </span>
                        </div>
                    </div>
                </header>

                {/* Page Content */}
                <main className="flex-1 p-6 overflow-y-auto scroll-smooth">
                    <div className="max-w-7xl mx-auto animate-fade-in">
                        {children}
                    </div>
                </main>
            </div>

            {/* Overlay for mobile sidebar */}
            {isSidebarOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-40 lg:hidden backdrop-blur-sm"
                    onClick={() => setIsSidebarOpen(false)}
                />
            )}
        </div>
    );
};

export default Layout;
