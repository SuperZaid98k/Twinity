import { Bot, Globe, Lock, MoreVertical, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

interface CloneCardProps {
    id: string;
    name: string;
    description: string;
    isPublic: boolean;
    status: "active" | "training" | "draft";
    conversations: number;
    onToggleVisibility: (id: string) => void;
}

const statusColors = {
    active: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    training: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    draft: "bg-muted text-muted-foreground border-border",
};

const CloneCard = ({ id, name, description, isPublic, status, conversations, onToggleVisibility }: CloneCardProps) => {
    return (
        <div className="glass-hover rounded-xl p-5 flex flex-col gap-4 group">
            <div className="flex items-start justify-between">
                <div className="w-12 h-12 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                    <Bot className="w-6 h-6 text-primary" />
                </div>
                <div className="flex items-center gap-2">
                    <span className={`text-xs px-2.5 py-1 rounded-full border ${statusColors[status]}`}>
                        {status}
                    </span>
                    <button className="text-muted-foreground hover:text-foreground transition-colors">
                        <MoreVertical className="w-4 h-4" />
                    </button>
                </div>
            </div>

            <div>
                <h3 className="font-display font-semibold text-foreground text-lg">{name}</h3>
                <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{description}</p>
            </div>

            <div className="flex items-center justify-between mt-auto pt-3 border-t border-border/50">
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <button
                        onClick={() => onToggleVisibility(id)}
                        className="flex items-center gap-1 hover:text-primary transition-colors"
                    >
                        {isPublic ? <Globe className="w-3.5 h-3.5" /> : <Lock className="w-3.5 h-3.5" />}
                        {isPublic ? "Public" : "Private"}
                    </button>
                    <span className="flex items-center gap-1">
                        <MessageSquare className="w-3.5 h-3.5" />
                        {conversations}
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <Link to={`/chat/${id}`}>
                        <Button size="sm" className="bg-primary text-primary-foreground hover:bg-primary/90 text-xs h-8">
                            Chat
                        </Button>
                    </Link>
                    <Link to={`/clone/${id}`}>
                        <Button size="sm" variant="ghost" className="text-primary hover:bg-primary/10 text-xs h-8">
                            Manage
                        </Button>
                    </Link>
                </div>
            </div>
        </div>
    );
};

export default CloneCard;
