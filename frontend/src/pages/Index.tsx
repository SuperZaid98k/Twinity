import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import Header from "@/components/Header";
import { Bot, Sparkles, Shield, Zap, Users, ArrowRight } from "lucide-react";
import heroBg from "@/assets/hero-bg.jpg";

const features = [
    {
        icon: Bot,
        title: "AI Clone Creation",
        description: "Build a digital version of yourself that thinks, speaks, and responds just like you.",
    },
    {
        icon: Sparkles,
        title: "Multi-Modal Training",
        description: "Upload voice samples, face data, and documents to train your AI twin.",
    },
    {
        icon: Shield,
        title: "Privacy Control",
        description: "Choose to keep your clone private or share it publicly with the world.",
    },
    {
        icon: Zap,
        title: "Instant Deployment",
        description: "Your AI clone is ready to chat in minutes, not days.",
    },
];

const Index = () => {
    return (
        <div className="min-h-screen bg-background">
            <Header />

            {/* Hero */}
            <section className="relative pt-32 pb-20 overflow-hidden">
                <div
                    className="absolute inset-0 opacity-30"
                    style={{ backgroundImage: `url(${heroBg})`, backgroundSize: "cover", backgroundPosition: "center" }}
                />
                <div className="absolute inset-0 bg-gradient-to-b from-background/60 via-background/80 to-background" />

                <div className="container mx-auto px-4 relative z-10 text-center max-w-4xl">
                    <div className="inline-flex items-center gap-2 glass rounded-full px-4 py-1.5 mb-8">
                        <Sparkles className="w-4 h-4 text-primary animate-pulse-glow" />
                        <span className="text-sm text-muted-foreground">Create Your Digital Twin</span>
                    </div>

                    <h1 className="text-5xl md:text-7xl font-display font-bold text-foreground leading-tight mb-6">
                        Your AI Clone,<br />
                        <span className="text-gradient">Your Digital Twin</span>
                    </h1>

                    <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
                        Build an AI version of yourself that can chat, answer questions, and interact — trained on your voice, face, and knowledge.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <Link to="/signup">
                            <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90 px-8 h-12 text-base font-display">
                                Create Your Clone
                                <ArrowRight className="w-4 h-4 ml-2" />
                            </Button>
                        </Link>
                        <Link to="/signin">
                            <Button size="lg" variant="outline" className="border-border hover:bg-secondary h-12 px-8 text-base">
                                Sign In
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Features */}
            <section className="py-20 relative">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-14">
                        <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">
                            Everything You Need
                        </h2>
                        <p className="text-muted-foreground max-w-lg mx-auto">
                            From training data to deployment, Twinity handles it all.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 max-w-6xl mx-auto">
                        {features.map((f) => (
                            <div key={f.title} className="glass-hover rounded-xl p-6 text-center group">
                                <div className="w-12 h-12 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-4 group-hover:glow-box transition-all">
                                    <f.icon className="w-6 h-6 text-primary" />
                                </div>
                                <h3 className="font-display font-semibold text-foreground mb-2">{f.title}</h3>
                                <p className="text-sm text-muted-foreground">{f.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className="py-20">
                <div className="container mx-auto px-4">
                    <div className="glass glow-box rounded-2xl p-10 md:p-16 text-center max-w-3xl mx-auto">
                        <Users className="w-10 h-10 text-primary mx-auto mb-6" />
                        <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">
                            Ready to Meet Your Twin?
                        </h2>
                        <p className="text-muted-foreground mb-8 max-w-md mx-auto">
                            Join thousands creating their digital clones on Twinity.
                        </p>
                        <Link to="/signup">
                            <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90 px-8 h-12">
                                Get Started Free
                                <ArrowRight className="w-4 h-4 ml-2" />
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="border-t border-border/30 py-8">
                <div className="container mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <Bot className="w-5 h-5 text-primary" />
                        <span className="font-display font-semibold text-foreground">Twinity</span>
                    </div>
                    <p className="text-sm text-muted-foreground">© 2026 Twinity. All rights reserved.</p>
                </div>
            </footer>
        </div>
    );
};

export default Index;