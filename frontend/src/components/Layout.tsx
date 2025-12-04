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
    Users,
    Search,
    LogOut,
    User
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
    const [isProfileOpen, setIsProfileOpen] = useState(false);
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
        <div className="min-h-screen bg-background text-text-primary flex font-sans overflow-hidden">
            {/* Sidebar */}
            <aside
                className={cn(
                    "fixed inset-y-0 left-0 z-50 w-72 glass transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 border-r border-border",
                    isSidebarOpen ? "translate-x-0" : "-translate-x-full"
                )}
            >
                <div className="h-full flex flex-col">
                    {/* Logo */}
                    <div className="h-20 flex items-center px-8 border-b border-border">
                        <div className="p-2 bg-primary/10 rounded-xl mr-3">
                            <Activity className="h-6 w-6 text-primary" />
                        </div>
                        <span className="text-2xl font-bold tracking-tight">
                            NoShow<span className="text-primary">Predict</span>
                        </span>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 px-4 py-8 space-y-2 overflow-y-auto">
                        <div className="px-4 mb-2 text-xs font-semibold text-text-tertiary uppercase tracking-wider">
                            Main Menu
                        </div>
                        {navigation.map((item) => {
                            const isActive = location.pathname === item.href;
                            return (
                                <Link
                                    key={item.name}
                                    to={item.href}
                                    className={cn(
                                        "flex items-center px-4 py-3.5 rounded-xl transition-all duration-200 group relative overflow-hidden",
                                        isActive
                                            ? "bg-primary/10 text-primary shadow-[0_0_20px_rgba(59,130,246,0.15)]"
                                            : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
                                    )}
                                >
                                    {isActive && (
                                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-primary rounded-r-full" />
                                    )}
                                    <item.icon className={cn(
                                        "h-5 w-5 mr-3 transition-colors",
                                        isActive ? "text-primary" : "text-text-tertiary group-hover:text-text-primary"
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
                    <div className="p-4 border-t border-border bg-surface/30">
                        <div className="flex items-center p-3 rounded-xl hover:bg-surface-hover transition-colors cursor-pointer group">
                            <div className="h-10 w-10 rounded-full bg-gradient-to-tr from-primary to-accent flex items-center justify-center text-white font-bold shadow-lg ring-2 ring-surface group-hover:ring-primary/50 transition-all">
                                DA
                            </div>
                            <div className="ml-3 flex-1 min-w-0">
                                <p className="text-sm font-semibold text-text-primary truncate">Dr. Admin</p>
                                <p className="text-xs text-text-secondary truncate">Administrator</p>
                            </div>
                            <Settings className="h-4 w-4 text-text-tertiary group-hover:text-primary transition-colors" />
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0 relative">
                {/* Header */}
                <header className="h-20 glass sticky top-0 z-40 px-8 flex items-center justify-between backdrop-blur-xl bg-background/80 border-b border-border">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                            className="lg:hidden p-2 rounded-xl text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                        >
                            {isSidebarOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                        </button>

                        {/* Search Bar */}
                        <div className="hidden md:flex items-center relative group">
                            <Search className="absolute left-3 h-4 w-4 text-text-tertiary group-focus-within:text-primary transition-colors" />
                            <input
                                type="text"
                                placeholder="Search patients, appointments..."
                                className="bg-surface/50 border border-border text-sm rounded-xl pl-10 pr-4 py-2.5 w-64 focus:w-80 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all text-text-primary placeholder:text-text-tertiary"
                            />
                        </div>
                    </div>

                    <div className="flex items-center ml-auto space-x-2">
                        <div className="relative">
                            <button
                                onClick={() => setIsNotificationsOpen(!isNotificationsOpen)}
                                className={cn(
                                    "p-2.5 rounded-xl text-text-secondary hover:text-primary hover:bg-primary/10 transition-all relative",
                                    isNotificationsOpen && "bg-primary/10 text-primary"
                                )}
                            >
                                <Bell className="h-5 w-5" />
                                <span className="absolute top-2 right-2 h-2 w-2 rounded-full bg-error ring-2 ring-background animate-pulse"></span>
                            </button>

                            {/* Notifications Dropdown */}
                            {isNotificationsOpen && (
                                <>
                                    <div className="fixed inset-0 z-40" onClick={() => setIsNotificationsOpen(false)} />
                                    <div className="absolute right-0 mt-4 w-96 glass-card rounded-2xl shadow-2xl overflow-hidden z-50 animate-fade-in origin-top-right ring-1 ring-border">
                                        <div className="p-4 border-b border-border flex justify-between items-center bg-surface/50">
                                            <h3 className="font-bold text-text-primary">Notifications</h3>
                                            <button className="text-xs text-primary font-medium hover:text-primary-hover transition-colors">
                                                Mark all as read
                                            </button>
                                        </div>
                                        <div className="max-h-[400px] overflow-y-auto custom-scrollbar">
                                            <div className="p-4 border-b border-border hover:bg-surface-hover/50 transition-colors cursor-pointer relative group">
                                                <div className="absolute left-0 top-0 bottom-0 w-1 bg-error opacity-0 group-hover:opacity-100 transition-opacity" />
                                                <div className="flex items-start gap-4">
                                                    <div className="p-2.5 bg-error/10 rounded-xl shrink-0">
                                                        <Activity className="w-5 h-5 text-error" />
                                                    </div>
                                                    <div>
                                                        <p className="text-sm font-semibold text-text-primary">High Risk Patient Detected</p>
                                                        <p className="text-xs text-text-secondary mt-1 leading-relaxed">Patient #12345 flagged as <span className="text-error font-medium">CRITICAL</span> risk. Immediate action recommended.</p>
                                                        <p className="text-[10px] text-text-tertiary mt-2 font-medium uppercase tracking-wide">2 mins ago</p>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="p-4 hover:bg-surface-hover/50 transition-colors cursor-pointer relative group">
                                                <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary opacity-0 group-hover:opacity-100 transition-opacity" />
                                                <div className="flex items-start gap-4">
                                                    <div className="p-2.5 bg-primary/10 rounded-xl shrink-0">
                                                        <Upload className="w-5 h-5 text-primary" />
                                                    </div>
                                                    <div>
                                                        <p className="text-sm font-semibold text-text-primary">Batch Upload Complete</p>
                                                        <p className="text-xs text-text-secondary mt-1 leading-relaxed">Successfully processed 150 records from <code>appointments_q3.csv</code>.</p>
                                                        <p className="text-[10px] text-text-tertiary mt-2 font-medium uppercase tracking-wide">1 hour ago</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="p-3 text-center border-t border-border bg-surface/50 backdrop-blur-sm">
                                            <Link
                                                to="/patients"
                                                className="text-sm text-primary hover:text-primary-hover font-medium flex items-center justify-center gap-1 group"
                                                onClick={() => setIsNotificationsOpen(false)}
                                            >
                                                View All Alerts
                                                <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                                            </Link>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>

                        <div className="h-8 w-px bg-border mx-2"></div>

                        {/* Profile Dropdown Trigger */}
                        <div className="relative">
                            <button
                                onClick={() => setIsProfileOpen(!isProfileOpen)}
                                className="flex items-center gap-3 p-1.5 pr-3 rounded-xl hover:bg-surface-hover transition-colors"
                            >
                                <div className="h-8 w-8 rounded-full bg-gradient-to-tr from-primary to-accent flex items-center justify-center text-white text-xs font-bold shadow-md ring-2 ring-background">
                                    DA
                                </div>
                                <span className="text-sm font-medium text-text-primary hidden sm:block">Dr. Admin</span>
                                <ChevronRight className={cn("w-4 h-4 text-text-tertiary transition-transform duration-200", isProfileOpen && "rotate-90")} />
                            </button>

                            {isProfileOpen && (
                                <>
                                    <div className="fixed inset-0 z-40" onClick={() => setIsProfileOpen(false)} />
                                    <div className="absolute right-0 mt-2 w-56 glass-card rounded-xl shadow-xl overflow-hidden z-50 animate-fade-in ring-1 ring-border">
                                        <div className="p-2 space-y-1">
                                            <Link to="/profile" className="flex items-center gap-3 px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-surface-hover rounded-lg transition-colors" onClick={() => setIsProfileOpen(false)}>
                                                <User className="w-4 h-4" />
                                                My Profile
                                            </Link>
                                            <Link to="/settings" className="flex items-center gap-3 px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-surface-hover rounded-lg transition-colors" onClick={() => setIsProfileOpen(false)}>
                                                <Settings className="w-4 h-4" />
                                                Settings
                                            </Link>
                                            <div className="h-px bg-border my-1" />
                                            <button className="w-full flex items-center gap-3 px-3 py-2 text-sm text-error hover:bg-error/10 rounded-lg transition-colors">
                                                <LogOut className="w-4 h-4" />
                                                Sign Out
                                            </button>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </header>

                {/* Page Content */}
                <main className="flex-1 p-6 lg:p-8 overflow-y-auto scroll-smooth custom-scrollbar">
                    <div className="max-w-7xl mx-auto animate-fade-in space-y-8">
                        {children}
                    </div>
                </main>
            </div>

            {/* Overlay for mobile sidebar */}
            {isSidebarOpen && (
                <div
                    className="fixed inset-0 bg-black/60 z-40 lg:hidden backdrop-blur-sm transition-opacity"
                    onClick={() => setIsSidebarOpen(false)}
                />
            )}
        </div>
    );
};

export default Layout;
