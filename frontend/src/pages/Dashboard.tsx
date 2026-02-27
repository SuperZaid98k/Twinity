import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import Header from "@/components/Header";
import CloneCard from "@/components/CloneCard";
import { Plus, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { getMyClones, updateCloneVisibility } from "@/lib/api";
import { toast } from "sonner";

interface Clone {
    id: string;
    name: string;
    description: string;
    isPublic: boolean;
    status: "active" | "training" | "draft";
    conversations: number;
}

const Dashboard = () => {
    const [clones, setClones] = useState<Clone[]>([]);
    const [search, setSearch] = useState("");
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        let mounted = true;
        const loadClones = async () => {
            try {
                const response = await getMyClones();
                if (!mounted) return;
                const mapped: Clone[] = response.clones.map((clone) => ({
                    id: clone.id,
                    name: clone.name,
                    description: clone.description,
                    isPublic: clone.is_public,
                    status: "active",
                    conversations: clone.chunk_count ?? 0,
                }));
                setClones(mapped);
            } catch (error) {
                const message = error instanceof Error ? error.message : "Failed to load clones";
                toast.error(message);
            } finally {
                if (mounted) setIsLoading(false);
            }
        };

        loadClones();
        return () => {
            mounted = false;
        };
    }, []);

    const toggleVisibility = async (id: string) => {
        const current = clones.find((clone) => clone.id === id);
        if (!current) return;

        try {
            const response = await updateCloneVisibility(id, !current.isPublic);
            setClones((prev) =>
                prev.map((clone) =>
                    clone.id === id ? { ...clone, isPublic: response.clone.is_public } : clone
                )
            );
        } catch (error) {
            const message = error instanceof Error ? error.message : "Failed to update visibility";
            toast.error(message);
        }
    };

    const filtered = clones.filter(c =>
        c.name.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="min-h-screen bg-background">
            <Header />
            <main className="container mx-auto px-4 pt-24 pb-16">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
                    <div>
                        <h1 className="text-3xl font-display font-bold text-foreground">My Clones</h1>
                        <p className="text-muted-foreground text-sm mt-1">{clones.length} AI clones created</p>
                    </div>
                    <Link to="/create">
                        <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                            <Plus className="w-4 h-4 mr-2" />
                            New Clone
                        </Button>
                    </Link>
                </div>

                <div className="relative max-w-sm mb-6">
                    <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
                    <Input
                        placeholder="Search clones..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="pl-10 bg-muted/50 border-border focus:border-primary"
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                    {!isLoading && filtered.length === 0 && (
                        <div className="col-span-full glass rounded-xl p-6 text-sm text-muted-foreground">
                            No clones yet. Create your first clone to get started.
                        </div>
                    )}
                    {filtered.map(clone => (
                        <CloneCard
                            key={clone.id}
                            {...clone}
                            onToggleVisibility={toggleVisibility}
                        />
                    ))}
                    <Link to="/create" className="glass-hover rounded-xl p-5 flex flex-col items-center justify-center gap-3 min-h-[200px] group border-dashed border-2 border-border hover:border-primary/30">
                        <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                            <Plus className="w-6 h-6 text-muted-foreground group-hover:text-primary transition-colors" />
                        </div>
                        <span className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">Create New Clone</span>
                    </Link>
                </div>
            </main>
        </div>
    );
};

export default Dashboard;
