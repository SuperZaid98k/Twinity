import { Upload, X, FileText, Mic, Camera } from "lucide-react";
import { useState, useRef } from "react";

interface UploadZoneProps {
    type: "voice" | "face" | "documents";
    title: string;
    description: string;
    accept: string;
    multiple?: boolean;
}

const icons = {
    voice: Mic,
    face: Camera,
    documents: FileText,
};

const UploadZone = ({ type, title, description, accept, multiple = false }: UploadZoneProps) => {
    const [files, setFiles] = useState<File[]>([]);
    const [dragOver, setDragOver] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const Icon = icons[type];

    const handleFiles = (newFiles: FileList | null) => {
        if (!newFiles) return;
        setFiles(prev => [...prev, ...Array.from(newFiles)]);
    };

    const removeFile = (index: number) => {
        setFiles(prev => prev.filter((_, i) => i !== index));
    };

    return (
        <div className="glass rounded-xl p-6 space-y-4">
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
                    <Icon className="w-5 h-5 text-primary" />
                </div>
                <div>
                    <h3 className="font-display font-semibold text-foreground">{title}</h3>
                    <p className="text-xs text-muted-foreground">{description}</p>
                </div>
            </div>

            <div
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/40"
                    }`}
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
            >
                <Upload className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">
                    Drop files here or <span className="text-primary">browse</span>
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">{accept}</p>
                <input
                    ref={inputRef}
                    type="file"
                    accept={accept}
                    multiple={multiple}
                    className="hidden"
                    onChange={(e) => handleFiles(e.target.files)}
                />
            </div>

            {files.length > 0 && (
                <div className="space-y-2">
                    {files.map((file, i) => (
                        <div key={i} className="flex items-center justify-between bg-muted/50 rounded-lg px-3 py-2">
                            <div className="flex items-center gap-2 min-w-0">
                                <FileText className="w-4 h-4 text-primary shrink-0" />
                                <span className="text-sm text-foreground truncate">{file.name}</span>
                                <span className="text-xs text-muted-foreground shrink-0">
                                    {(file.size / 1024 / 1024).toFixed(1)} MB
                                </span>
                            </div>
                            <button onClick={() => removeFile(i)} className="text-muted-foreground hover:text-destructive transition-colors shrink-0 ml-2">
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default UploadZone;