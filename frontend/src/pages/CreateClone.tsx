import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import Header from "@/components/Header";
import { Bot, Globe, Lock, ArrowLeft, ArrowRight, Check, FileText, Mic, Camera } from "lucide-react";
import { toast } from "sonner";
import { createClone } from "@/lib/api";

const steps = ["Details", "Training Data", "Visibility"];

const CreateClone = () => {
    const [step, setStep] = useState(0);
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [isPublic, setIsPublic] = useState(false);
    const navigate = useNavigate();

    const handleCreate = async () => {
        if (!name) {
            toast.error("Please provide a name for your clone");
            return;
        }
        try {
            await createClone(name, description, isPublic);
            toast.success("Clone created successfully!");
            navigate("/dashboard");
        } catch (error) {
            const message = error instanceof Error ? error.message : "Failed to create clone";
            toast.error(message);
        }
    };

    return (
        <div className="min-h-screen bg-background">
            <Header />
            <main className="container mx-auto px-4 pt-24 pb-16 max-w-3xl">
                <button onClick={() => navigate("/dashboard")} className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6">
                    <ArrowLeft className="w-4 h-4" />
                    Back to Dashboard
                </button>

                <h1 className="text-3xl font-display font-bold text-foreground mb-2">Create New Clone</h1>
                <p className="text-muted-foreground text-sm mb-8">Set up your AI twin in a few steps.</p>

                {/* Steps indicator */}
                <div className="flex items-center gap-2 mb-10">
                    {steps.map((s, i) => (
                        <div key={s} className="flex items-center gap-2">
                            <button
                                onClick={() => i < step && setStep(i)}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all ${i === step
                                    ? "bg-primary/10 text-primary border border-primary/30"
                                    : i < step
                                        ? "bg-primary/5 text-primary/60"
                                        : "text-muted-foreground"
                                    }`}
                            >
                                {i < step ? <Check className="w-3.5 h-3.5" /> : <span className="w-5 h-5 rounded-full border border-current flex items-center justify-center text-xs">{i + 1}</span>}
                                <span className="hidden sm:inline">{s}</span>
                            </button>
                            {i < steps.length - 1 && <div className="w-8 h-px bg-border" />}
                        </div>
                    ))}
                </div>

                {/* Step 1: Details */}
                {step === 0 && (
                    <div className="glass rounded-xl p-6 space-y-5">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
                                <Bot className="w-5 h-5 text-primary" />
                            </div>
                            <h2 className="font-display font-semibold text-foreground text-lg">Clone Details</h2>
                        </div>

                        <div className="space-y-2">
                            <Label className="text-sm text-muted-foreground">Clone Name</Label>
                            <Input
                                placeholder="e.g. Professional Me"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                className="bg-muted/50 border-border focus:border-primary"
                            />
                        </div>

                        <div className="space-y-2">
                            <Label className="text-sm text-muted-foreground">Description</Label>
                            <Textarea
                                placeholder="Describe what this clone will be used for..."
                                value={description}
                                onChange={(e) => setDescription(e.target.value)}
                                className="bg-muted/50 border-border focus:border-primary min-h-[100px] resize-none"
                            />
                        </div>
                    </div>
                )}

                {/* Step 2: Knowledge Setup */}
                {step === 1 && (
                    <div className="glass rounded-xl p-6 space-y-6">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
                                <FileText className="w-5 h-5 text-primary" />
                            </div>
                            <div>
                                <h2 className="font-display font-semibold text-foreground text-lg">Knowledge Base First</h2>
                                <p className="text-xs text-muted-foreground">Add training data after clone creation in Manage Knowledge Base.</p>
                            </div>
                        </div>

                        <div className="rounded-lg border border-border/50 bg-muted/30 p-4 text-sm text-muted-foreground">
                            For now, this clone is trained using text and supported files (PDF, DOCX, PPTX, TXT, CSV, XLSX, images).
                            After creation, open <span className="text-foreground font-medium">Manage</span> from your dashboard to upload data chunks.
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="rounded-lg border border-border/50 bg-muted/30 p-4">
                                <div className="flex items-center gap-2 text-foreground font-medium">
                                    <Mic className="w-4 h-4 text-muted-foreground" />
                                    Voice Training
                                </div>
                                <p className="text-xs text-muted-foreground mt-1">Coming soon.</p>
                            </div>
                            <div className="rounded-lg border border-border/50 bg-muted/30 p-4">
                                <div className="flex items-center gap-2 text-foreground font-medium">
                                    <Camera className="w-4 h-4 text-muted-foreground" />
                                    Face Training
                                </div>
                                <p className="text-xs text-muted-foreground mt-1">Coming soon.</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Step 3: Visibility */}
                {step === 2 && (
                    <div className="glass rounded-xl p-6 space-y-6">
                        <h2 className="font-display font-semibold text-foreground text-lg">Clone Visibility</h2>
                        <p className="text-sm text-muted-foreground">Choose who can interact with your AI clone. You can change this anytime.</p>

                        <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/30 border border-border">
                            <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                    {isPublic ? <Globe className="w-4 h-4 text-primary" /> : <Lock className="w-4 h-4 text-muted-foreground" />}
                                    <span className="font-display font-semibold text-foreground">
                                        {isPublic ? "Public" : "Private"}
                                    </span>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                    {isPublic
                                        ? "Anyone can discover and chat with your clone."
                                        : "Only you can access and chat with your clone."}
                                </p>
                            </div>
                            <Switch checked={isPublic} onCheckedChange={setIsPublic} />
                        </div>
                    </div>
                )}

                {/* Navigation */}
                <div className="flex items-center justify-between mt-8">
                    <Button
                        variant="ghost"
                        onClick={() => setStep(Math.max(0, step - 1))}
                        disabled={step === 0}
                        className="text-muted-foreground"
                    >
                        <ArrowLeft className="w-4 h-4 mr-2" />
                        Back
                    </Button>

                    {step < steps.length - 1 ? (
                        <Button
                            onClick={() => setStep(step + 1)}
                            className="bg-primary text-primary-foreground hover:bg-primary/90"
                        >
                            Next
                            <ArrowRight className="w-4 h-4 ml-2" />
                        </Button>
                    ) : (
                        <Button
                            onClick={handleCreate}
                            className="bg-primary text-primary-foreground hover:bg-primary/90"
                        >
                            <Check className="w-4 h-4 mr-2" />
                            Create Clone
                        </Button>
                    )}
                </div>
            </main>
        </div>
    );
};

export default CreateClone;
