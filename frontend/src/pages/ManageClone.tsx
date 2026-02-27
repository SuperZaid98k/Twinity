import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import Header from "@/components/Header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  ApiChunk,
  ApiChunkPreview,
  ApiClone,
  ApiDocument,
  deleteChunks,
  deleteClone,
  getClone,
  getCloneChunks,
  getCloneDocuments,
  previewFileChunks,
  previewTextChunks,
  updateCloneVisibility,
  updateCloneWhatsapp,
  uploadDocument,
  uploadSelectedChunks,
  uploadTextChunks,
} from "@/lib/api";
import { ArrowLeft, FileText, Link as LinkIcon, Phone, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";

const ManageClone = () => {
  const { cloneId } = useParams<{ cloneId: string }>();
  const navigate = useNavigate();

  const [clone, setClone] = useState<ApiClone | null>(null);
  const [chunks, setChunks] = useState<ApiChunk[]>([]);
  const [documents, setDocuments] = useState<ApiDocument[]>([]);
  const [textContent, setTextContent] = useState("");
  const [fileToPreview, setFileToPreview] = useState<File | null>(null);
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [documentDescription, setDocumentDescription] = useState("");
  const [previewChunks, setPreviewChunks] = useState<ApiChunkPreview[]>([]);
  const [selectedPreviewIds, setSelectedPreviewIds] = useState<number[]>([]);
  const [whatsappEnabled, setWhatsappEnabled] = useState(false);
  const [phoneNumber, setPhoneNumber] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSavingVisibility, setIsSavingVisibility] = useState(false);
  const [isSavingWhatsapp, setIsSavingWhatsapp] = useState(false);
  const [isUploadingText, setIsUploadingText] = useState(false);
  const [isPreviewingText, setIsPreviewingText] = useState(false);
  const [isPreviewingFile, setIsPreviewingFile] = useState(false);
  const [isUploadingPreviewSelection, setIsUploadingPreviewSelection] = useState(false);
  const [isUploadingDocument, setIsUploadingDocument] = useState(false);
  const [isDeletingClone, setIsDeletingClone] = useState(false);

  const selectedChunkCount = useMemo(() => selectedPreviewIds.length, [selectedPreviewIds]);

  const loadAll = async () => {
    if (!cloneId) return;
    setIsLoading(true);
    try {
      const [cloneResp, chunksResp, docsResp] = await Promise.all([
        getClone(cloneId),
        getCloneChunks(cloneId),
        getCloneDocuments(cloneId),
      ]);
      setClone(cloneResp.clone);
      setChunks(chunksResp.chunks);
      setDocuments(docsResp.documents);
      const hasWhatsapp = !!cloneResp.clone.phone_number;
      setWhatsappEnabled(hasWhatsapp);
      setPhoneNumber(cloneResp.clone.phone_number || "");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load clone data";
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadAll();
  }, [cloneId]);

  const refreshChunksAndDocs = async () => {
    if (!cloneId) return;
    const [chunksResp, docsResp] = await Promise.all([getCloneChunks(cloneId), getCloneDocuments(cloneId)]);
    setChunks(chunksResp.chunks);
    setDocuments(docsResp.documents);
  };

  const handleToggleVisibility = async () => {
    if (!cloneId || !clone) return;
    setIsSavingVisibility(true);
    try {
      const response = await updateCloneVisibility(cloneId, !clone.is_public);
      setClone(response.clone);
      toast.success(`Clone is now ${response.clone.is_public ? "public" : "private"}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to update visibility";
      toast.error(message);
    } finally {
      setIsSavingVisibility(false);
    }
  };

  const handleSaveWhatsapp = async () => {
    if (!cloneId) return;
    setIsSavingWhatsapp(true);
    try {
      const response = await updateCloneWhatsapp(cloneId, whatsappEnabled, phoneNumber);
      setClone(response.clone);
      setWhatsappEnabled(!!response.clone.phone_number);
      setPhoneNumber(response.clone.phone_number || "");
      toast.success("WhatsApp settings updated");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to update WhatsApp settings";
      toast.error(message);
    } finally {
      setIsSavingWhatsapp(false);
    }
  };

  const handleUploadTextDirect = async (e: FormEvent) => {
    e.preventDefault();
    if (!cloneId || !textContent.trim()) {
      toast.error("Enter text before uploading");
      return;
    }
    setIsUploadingText(true);
    try {
      const response = await uploadTextChunks(cloneId, textContent.trim());
      toast.success(`Added ${response.chunks_added} chunks`);
      setTextContent("");
      await refreshChunksAndDocs();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Text upload failed";
      toast.error(message);
    } finally {
      setIsUploadingText(false);
    }
  };

  const handlePreviewText = async () => {
    if (!cloneId || !textContent.trim()) {
      toast.error("Enter text to preview chunks");
      return;
    }
    setIsPreviewingText(true);
    try {
      const response = await previewTextChunks(cloneId, textContent.trim());
      setPreviewChunks(response.chunks);
      setSelectedPreviewIds(response.chunks.map((chunk) => chunk.id));
      toast.success(`Previewed ${response.total_chunks} chunks`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to preview text chunks";
      toast.error(message);
    } finally {
      setIsPreviewingText(false);
    }
  };

  const handlePreviewFile = async (e: FormEvent) => {
    e.preventDefault();
    if (!cloneId || !fileToPreview) {
      toast.error("Choose a file to preview");
      return;
    }
    setIsPreviewingFile(true);
    try {
      const response = await previewFileChunks(cloneId, fileToPreview);
      setPreviewChunks(response.chunks);
      setSelectedPreviewIds(response.chunks.map((chunk) => chunk.id));
      toast.success(`Previewed ${response.total_chunks} chunks from file`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to preview file chunks";
      toast.error(message);
    } finally {
      setIsPreviewingFile(false);
    }
  };

  const togglePreviewSelection = (chunkId: number) => {
    setSelectedPreviewIds((prev) => (prev.includes(chunkId) ? prev.filter((id) => id !== chunkId) : [...prev, chunkId]));
  };

  const handleUploadSelectedPreview = async () => {
    if (!cloneId || selectedPreviewIds.length === 0) {
      toast.error("Select at least one chunk");
      return;
    }
    setIsUploadingPreviewSelection(true);
    try {
      const response = await uploadSelectedChunks(cloneId, selectedPreviewIds);
      toast.success(`Uploaded ${response.chunks_added} selected chunks`);
      setPreviewChunks([]);
      setSelectedPreviewIds([]);
      await refreshChunksAndDocs();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to upload selected chunks";
      toast.error(message);
    } finally {
      setIsUploadingPreviewSelection(false);
    }
  };

  const handleUploadDocument = async (e: FormEvent) => {
    e.preventDefault();
    if (!cloneId || !documentFile || !documentDescription.trim()) {
      toast.error("Add document and description");
      return;
    }
    setIsUploadingDocument(true);
    try {
      await uploadDocument(cloneId, documentDescription.trim(), documentFile);
      toast.success("Document uploaded");
      setDocumentFile(null);
      setDocumentDescription("");
      await refreshChunksAndDocs();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Document upload failed";
      toast.error(message);
    } finally {
      setIsUploadingDocument(false);
    }
  };

  const handleDeleteChunk = async (chunkId: string) => {
    if (!cloneId) return;
    try {
      await deleteChunks(cloneId, [chunkId]);
      setChunks((prev) => prev.filter((chunk) => chunk.id !== chunkId));
      toast.success("Chunk deleted");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to delete chunk";
      toast.error(message);
    }
  };

  const handleDeleteClone = async () => {
    if (!cloneId) return;
    if (!window.confirm("Delete this clone and all its data?")) return;
    setIsDeletingClone(true);
    try {
      await deleteClone(cloneId);
      toast.success("Clone deleted");
      navigate("/dashboard");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to delete clone";
      toast.error(message);
    } finally {
      setIsDeletingClone(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 pt-24 pb-16 max-w-6xl space-y-6">
        <Link to="/dashboard" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </Link>

        <section className="glass rounded-xl p-6 border border-border/50">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div>
              <h1 className="text-2xl font-display font-bold text-foreground">{clone?.name || "Manage Clone"}</h1>
              <p className="text-sm text-muted-foreground mt-1">{clone?.description || "Configure and train your clone from this page."}</p>
            </div>
            <div className="flex items-center gap-2">
              <Link to={`/chat/${cloneId}`}>
                <Button variant="outline">Open Chat</Button>
              </Link>
              <Button variant="outline" onClick={() => void loadAll()} className="gap-2">
                <RefreshCw className="w-4 h-4" />
                Refresh
              </Button>
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="glass rounded-xl p-6 border border-border/50 space-y-4">
            <h2 className="font-display font-semibold text-foreground">Clone Settings</h2>
            {isLoading ? (
              <p className="text-sm text-muted-foreground">Loading...</p>
            ) : (
              <>
                <div className="flex items-center justify-between rounded-lg border border-border/50 bg-muted/30 p-4">
                  <div>
                    <p className="text-sm font-medium text-foreground">Public visibility</p>
                    <p className="text-xs text-muted-foreground">Allow others to discover and chat with this clone.</p>
                  </div>
                  <Switch checked={!!clone?.is_public} onCheckedChange={() => void handleToggleVisibility()} disabled={isSavingVisibility} />
                </div>
                <div className="rounded-lg border border-border/50 bg-muted/30 p-4">
                  <p className="text-sm font-medium text-foreground">Danger Zone</p>
                  <p className="text-xs text-muted-foreground mt-1">Delete this clone and all associated vectors/documents.</p>
                  <Button className="mt-3" variant="destructive" onClick={() => void handleDeleteClone()} disabled={isDeletingClone}>
                    {isDeletingClone ? "Deleting..." : "Delete Clone"}
                  </Button>
                </div>
              </>
            )}
          </div>

          <div className="glass rounded-xl p-6 border border-border/50 space-y-4">
            <h2 className="font-display font-semibold text-foreground">WhatsApp Integration</h2>
            <div className="flex items-center justify-between rounded-lg border border-border/50 bg-muted/30 p-4">
              <div className="flex items-center gap-2">
                <LinkIcon className="w-4 h-4 text-primary" />
                <div>
                  <p className="text-sm font-medium text-foreground">Enable updates via WhatsApp</p>
                  <p className="text-xs text-muted-foreground">Send messages/files to auto-update clone knowledge.</p>
                </div>
              </div>
              <Switch checked={whatsappEnabled} onCheckedChange={setWhatsappEnabled} />
            </div>
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground flex items-center gap-2">
                <Phone className="w-3.5 h-3.5" />
                Phone Number (E.164, e.g. +919876543210)
              </label>
              <Input value={phoneNumber} onChange={(e) => setPhoneNumber(e.target.value)} placeholder="+919876543210" />
            </div>
            <Button onClick={() => void handleSaveWhatsapp()} disabled={isSavingWhatsapp}>
              {isSavingWhatsapp ? "Saving..." : "Save WhatsApp Settings"}
            </Button>
          </div>
        </section>

        <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <form onSubmit={handleUploadTextDirect} className="glass rounded-xl p-6 border border-border/50 space-y-4">
            <h2 className="font-display font-semibold text-foreground">Knowledge Text</h2>
            <Textarea
              value={textContent}
              onChange={(e) => setTextContent(e.target.value)}
              placeholder="Paste the text you want this clone to learn..."
              className="min-h-[180px] bg-muted/50 border-border focus:border-primary resize-none"
            />
            <div className="flex gap-2 flex-wrap">
              <Button type="submit" disabled={isUploadingText}>
                {isUploadingText ? "Uploading..." : "Upload Text Directly"}
              </Button>
              <Button type="button" variant="outline" onClick={() => void handlePreviewText()} disabled={isPreviewingText}>
                {isPreviewingText ? "Previewing..." : "Preview Chunks"}
              </Button>
            </div>
          </form>

          <form onSubmit={handlePreviewFile} className="glass rounded-xl p-6 border border-border/50 space-y-4">
            <h2 className="font-display font-semibold text-foreground">Knowledge File</h2>
            <p className="text-xs text-muted-foreground">Preview chunks before storing vectors from file content.</p>
            <Input type="file" onChange={(e) => setFileToPreview(e.target.files?.[0] ?? null)} />
            <div className="flex gap-2 flex-wrap">
              <Button type="submit" disabled={isPreviewingFile || !fileToPreview}>
                {isPreviewingFile ? "Previewing..." : "Preview File Chunks"}
              </Button>
            </div>
          </form>
        </section>

        {previewChunks.length > 0 && (
          <section className="glass rounded-xl p-6 border border-border/50 space-y-4">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <h2 className="font-display font-semibold text-foreground">Chunk Preview ({previewChunks.length})</h2>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">{selectedChunkCount} selected</span>
                <Button onClick={() => void handleUploadSelectedPreview()} disabled={isUploadingPreviewSelection || selectedChunkCount === 0}>
                  {isUploadingPreviewSelection ? "Uploading..." : "Upload Selected Chunks"}
                </Button>
              </div>
            </div>
            <div className="space-y-2 max-h-[360px] overflow-y-auto">
              {previewChunks.map((chunk) => (
                <label key={chunk.id} className="flex items-start gap-3 rounded-lg border border-border/50 bg-muted/30 p-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedPreviewIds.includes(chunk.id)}
                    onChange={() => togglePreviewSelection(chunk.id)}
                    className="mt-1"
                  />
                  <div className="text-sm text-foreground whitespace-pre-wrap">{chunk.preview}</div>
                </label>
              ))}
            </div>
          </section>
        )}

        <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <form onSubmit={handleUploadDocument} className="glass rounded-xl p-6 border border-border/50 space-y-4">
            <h2 className="font-display font-semibold text-foreground">Attach Downloadable Document</h2>
            <p className="text-xs text-muted-foreground">Upload a file with description so it can be suggested/downloaded in chat.</p>
            <Input value={documentDescription} onChange={(e) => setDocumentDescription(e.target.value)} placeholder="Document description for retrieval..." />
            <Input type="file" onChange={(e) => setDocumentFile(e.target.files?.[0] ?? null)} />
            <Button type="submit" disabled={isUploadingDocument || !documentFile || !documentDescription.trim()}>
              {isUploadingDocument ? "Uploading..." : "Upload Document Attachment"}
            </Button>
          </form>

          <div className="glass rounded-xl p-6 border border-border/50 space-y-3">
            <h2 className="font-display font-semibold text-foreground">Attached Documents</h2>
            {documents.length === 0 ? (
              <p className="text-sm text-muted-foreground">No attached documents yet.</p>
            ) : (
              <div className="space-y-2 max-h-[220px] overflow-y-auto">
                {documents.map((doc) => (
                  <a key={doc.id} href={doc.download_url} target="_blank" rel="noreferrer" className="block rounded-lg border border-border/50 bg-muted/30 p-3 hover:bg-muted/40">
                    <div className="flex items-center gap-2 text-sm text-foreground">
                      <FileText className="w-4 h-4 text-primary" />
                      {doc.filename}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">{doc.description || "No description"}</div>
                  </a>
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="glass rounded-xl p-6 border border-border/50">
          <h2 className="font-display font-semibold text-foreground mb-4">Stored Chunks ({chunks.length})</h2>
          {chunks.length === 0 ? (
            <p className="text-sm text-muted-foreground">No chunks yet.</p>
          ) : (
            <div className="space-y-2 max-h-[360px] overflow-y-auto">
              {chunks.map((chunk) => (
                <div key={chunk.id} className="rounded-lg border border-border/50 bg-muted/30 p-3 flex items-start justify-between gap-3">
                  <p className="text-sm text-foreground whitespace-pre-wrap">{chunk.preview}</p>
                  <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" onClick={() => void handleDeleteChunk(chunk.id)}>
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default ManageClone;
