import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Header from "@/components/Header";
import { Bot, Lock, User } from "lucide-react";
import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";

const SignUp = () => {
    const [name, setName] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const navigate = useNavigate();
    const { signUp } = useAuth();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const ok = await signUp(name, username, password);
        if (ok) navigate("/dashboard");
    };

    return (
        <div className="min-h-screen bg-background flex flex-col">
            <Header />
            <div className="flex-1 flex items-center justify-center px-4 pt-16">
                <div className="w-full max-w-md">
                    <div className="text-center mb-8">
                        <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center mx-auto mb-4 glow-box">
                            <Bot className="w-7 h-7 text-primary" />
                        </div>
                        <h1 className="text-2xl font-display font-bold text-foreground">Create Your Account</h1>
                        <p className="text-muted-foreground text-sm mt-1">Start building your AI clone today</p>
                    </div>

                    <form onSubmit={handleSubmit} className="glass rounded-xl p-6 space-y-5">
                        <div className="space-y-2">
                            <Label htmlFor="name" className="text-sm text-muted-foreground">Full Name</Label>
                            <div className="relative">
                                <User className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
                                <Input id="name" placeholder="John Doe" value={name} onChange={(e) => setName(e.target.value)} className="pl-10 bg-muted/50 border-border focus:border-primary" />
                            </div>
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="username" className="text-sm text-muted-foreground">Username</Label>
                            <div className="relative">
                                <User className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
                                <Input id="username" placeholder="choose_a_username" value={username} onChange={(e) => setUsername(e.target.value)} className="pl-10 bg-muted/50 border-border focus:border-primary" />
                            </div>
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="password" className="text-sm text-muted-foreground">Password</Label>
                            <div className="relative">
                                <Lock className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
                                <Input id="password" type="password" placeholder="••••••••" value={password} onChange={(e) => setPassword(e.target.value)} className="pl-10 bg-muted/50 border-border focus:border-primary" />
                            </div>
                        </div>
                        <Button type="submit" className="w-full bg-primary text-primary-foreground hover:bg-primary/90 h-11">Create Account</Button>
                        <p className="text-center text-sm text-muted-foreground">
                            Already have an account? <Link to="/signin" className="text-primary hover:underline">Sign In</Link>
                        </p>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default SignUp;
