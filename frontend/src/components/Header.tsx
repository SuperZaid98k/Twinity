import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Bot, Menu, X, LogOut, Compass } from "lucide-react";
import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";

const Header = () => {
    const location = useLocation();
    const [mobileOpen, setMobileOpen] = useState(false);
    const isAuth = location.pathname === "/signin" || location.pathname === "/signup";
    const { isAuthenticated, user, signOut } = useAuth();

    return (
        <header className="fixed top-0 left-0 right-0 z-50 glass border-b border-border/30">
            <div className="container mx-auto flex items-center justify-between h-16 px-4">
                <Link to="/" className="flex items-center gap-2 group">
                    <div className="w-9 h-9 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center group-hover:glow-box transition-all">
                        <Bot className="w-5 h-5 text-primary" />
                    </div>
                    <span className="text-xl font-display font-bold text-foreground">
                        Twi<span className="text-primary">nity</span>
                    </span>
                </Link>

                {!isAuth && (
                    <>
                        <nav className="hidden md:flex items-center gap-6">
                            <Link to="/" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Home</Link>
                            <Link to="/explore" className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1">
                                <Compass className="w-3.5 h-3.5" />
                                Explore
                            </Link>
                            {isAuthenticated && (
                                <Link to="/dashboard" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Dashboard</Link>
                            )}
                        </nav>

                        <div className="hidden md:flex items-center gap-3">
                            {isAuthenticated ? (
                                <>
                                    <span className="text-sm text-muted-foreground">{user?.name}</span>
                                    <Button variant="ghost" size="sm" onClick={() => void signOut()} className="text-muted-foreground hover:text-foreground">
                                        <LogOut className="w-4 h-4 mr-1" />
                                        Sign Out
                                    </Button>
                                </>
                            ) : (
                                <>
                                    <Link to="/signin">
                                        <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">Sign In</Button>
                                    </Link>
                                    <Link to="/signup">
                                        <Button size="sm" className="bg-primary text-primary-foreground hover:bg-primary/90">Get Started</Button>
                                    </Link>
                                </>
                            )}
                        </div>

                        <button className="md:hidden text-muted-foreground" onClick={() => setMobileOpen(!mobileOpen)}>
                            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                        </button>
                    </>
                )}
            </div>

            {mobileOpen && (
                <div className="md:hidden glass border-t border-border/30 p-4 space-y-3">
                    <Link to="/" className="block text-sm text-muted-foreground" onClick={() => setMobileOpen(false)}>Home</Link>
                    <Link to="/explore" className="block text-sm text-muted-foreground" onClick={() => setMobileOpen(false)}>Explore</Link>
                    {isAuthenticated && (
                        <Link to="/dashboard" className="block text-sm text-muted-foreground" onClick={() => setMobileOpen(false)}>Dashboard</Link>
                    )}
                    <div className="flex gap-2 pt-2">
                        {isAuthenticated ? (
                            <Button variant="ghost" size="sm" onClick={() => { void signOut(); setMobileOpen(false); }}>
                                <LogOut className="w-4 h-4 mr-1" /> Sign Out
                            </Button>
                        ) : (
                            <>
                                <Link to="/signin"><Button variant="ghost" size="sm">Sign In</Button></Link>
                                <Link to="/signup"><Button size="sm" className="bg-primary text-primary-foreground">Get Started</Button></Link>
                            </>
                        )}
                    </div>
                </div>
            )}
        </header>
    );
};

export default Header;
