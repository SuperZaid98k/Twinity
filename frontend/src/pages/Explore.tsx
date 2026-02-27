import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import Header from "@/components/Header";
import { Search, Bot, Globe, MessageSquare, User } from "lucide-react";
import { getPublicClones } from "@/lib/api";
import { toast } from "sonner";
import { Link } from "react-router-dom";

interface PublicClone {
    id: string;
    name: string;
    description: string;
    creator: string;
    conversations: number;
    avatar?: string;
}

const Explore = () => {
    const [search, setSearch] = useState("");
    const [publicClones, setPublicClones] = useState<PublicClone[]>([]);

    useEffect(() => {
        let mounted = true;
        const loadPublicClones = async () => {
            try {
                const response = await getPublicClones();
                if (!mounted) return;
                setPublicClones(
                    response.clones.map((clone) => ({
                        id: clone.id,
                        name: clone.name,
                        description: clone.description,
                        creator: clone.owner_full_name || "Unknown creator",
                        conversations: clone.chunk_count ?? 0,
                    }))
                );
            } catch (error) {
                const message = error instanceof Error ? error.message : "Failed to load public clones";
                toast.error(message);
            }
        };

        loadPublicClones();
        return () => {
            mounted = false;
        };
    }, []);

    const filtered = publicClones.filter(
        (c) =>
            c.name.toLowerCase().includes(search.toLowerCase()) ||
            c.description.toLowerCase().includes(search.toLowerCase()) ||
            c.creator.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="min-h-screen bg-background">
            <Header />
            <main className="container mx-auto px-4 pt-24 pb-16">
                <div className="text-center mb-10">
                    <h1 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-3">
                        Explore Public Clones
                    </h1>
                    <p className="text-muted-foreground max-w-lg mx-auto">
                        Discover and chat with AI clones created by the community.
                    </p>
                </div>

                <div className="relative max-w-md mx-auto mb-10">
                    <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
                    <Input
                        placeholder="Search by name, topic, or creator..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="pl-10 bg-muted/50 border-border focus:border-primary"
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 max-w-6xl mx-auto">
                    {filtered.map((clone) => (
                        <div key={clone.id} className="glass-hover rounded-xl p-5 flex flex-col gap-4 group">
                            <div className="flex items-start gap-3">
                                <div className="w-12 h-12 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
                                    <Bot className="w-6 h-6 text-primary" />
                                </div>
                                <div className="min-w-0">
                                    <h3 className="font-display font-semibold text-foreground text-lg truncate">{clone.name}</h3>
                                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                        <User className="w-3 h-3" />
                                        {clone.creator}
                                    </div>
                                </div>
                            </div>

                            <p className="text-sm text-muted-foreground line-clamp-2">{clone.description}</p>

                            <div className="flex items-center justify-between mt-auto pt-3 border-t border-border/50">
                                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                                    <span className="flex items-center gap-1">
                                        <Globe className="w-3.5 h-3.5 text-primary" />
                                        Public
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <MessageSquare className="w-3.5 h-3.5" />
                                        {clone.conversations.toLocaleString()}
                                    </span>
                                </div>
                                <Link to={`/chat/${clone.id}`}>
                                    <Button size="sm" className="bg-primary text-primary-foreground hover:bg-primary/90 text-xs h-8">
                                        Chat Now
                                    </Button>
                                </Link>
                            </div>
                        </div>
                    ))}
                </div>

                {filtered.length === 0 && (
                    <div className="text-center py-16">
                        <Bot className="w-12 h-12 text-muted-foreground/40 mx-auto mb-4" />
                        <p className="text-muted-foreground">No clones found matching your search.</p>
                    </div>
                )}
            </main>
        </div>
    );
};

export default Explore;
