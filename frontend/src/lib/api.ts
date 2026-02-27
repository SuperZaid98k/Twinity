export interface ApiUser {
  name: string;
  username: string;
  email?: string;
}

export interface ApiClone {
  id: string;
  name: string;
  description: string;
  is_public: boolean;
  phone_number?: string;
  owner?: string;
  owner_full_name?: string;
  chunk_count: number;
  created_at?: string;
  last_updated?: string;
}

export interface ApiChunkPreview {
  id: number;
  text: string;
  preview: string;
  metadata: Record<string, unknown>;
  selected?: boolean;
  section?: string;
  type?: string;
}

export interface ApiChunk {
  id: string;
  text: string;
  preview: string;
  metadata: Record<string, unknown>;
}

export interface ApiDocument {
  id: string;
  filename: string;
  unique_filename: string;
  description: string;
  file_size?: number;
  download_url: string;
}

type RequestOptions = {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
};

async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(path, {
    method: options.method ?? "GET",
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "include",
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });

  const rawText = await response.text();
  const payload = (() => {
    try {
      return rawText ? JSON.parse(rawText) : {};
    } catch {
      return {};
    }
  })();

  if (!response.ok) {
    const message =
      payload?.error ||
      payload?.message ||
      (rawText && rawText.length < 300 ? rawText : "") ||
      `Request failed (${response.status})`;
    throw new Error(message);
  }

  return payload as T;
}

async function apiFormRequest<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    credentials: "include",
    body: formData,
  });

  const rawText = await response.text();
  const payload = (() => {
    try {
      return rawText ? JSON.parse(rawText) : {};
    } catch {
      return {};
    }
  })();
  if (!response.ok) {
    const message =
      payload?.error ||
      payload?.message ||
      (rawText && rawText.length < 300 ? rawText : "") ||
      `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload as T;
}

export async function getCurrentUser() {
  return apiRequest<{ authenticated: boolean; user: ApiUser }>("/api/auth/me");
}

export async function signIn(username: string, password: string) {
  return apiRequest<{ success: boolean; user: ApiUser }>("/api/auth/login", {
    method: "POST",
    body: { username, password },
  });
}

export async function signUp(name: string, username: string, password: string) {
  return apiRequest<{ success: boolean; user: ApiUser }>("/api/auth/register", {
    method: "POST",
    body: { name, username, password },
  });
}

export async function signOut() {
  return apiRequest<{ success: boolean }>("/api/auth/logout", { method: "POST" });
}

export async function getMyClones() {
  return apiRequest<{ clones: ApiClone[] }>("/api/clones");
}

export async function getPublicClones() {
  return apiRequest<{ clones: ApiClone[] }>("/api/clones/public");
}

export async function createClone(name: string, description: string, isPublic: boolean) {
  return apiRequest<{ success: boolean; clone: ApiClone }>("/api/clones", {
    method: "POST",
    body: { name, description, is_public: isPublic },
  });
}

export async function getClone(cloneId: string) {
  return apiRequest<{ clone: ApiClone }>(`/api/clones/${cloneId}`);
}

export async function updateCloneVisibility(cloneId: string, isPublic: boolean) {
  return apiRequest<{ success: boolean; clone: ApiClone }>(`/api/clones/${cloneId}/visibility`, {
    method: "PATCH",
    body: { is_public: isPublic },
  });
}

export async function updateCloneWhatsapp(cloneId: string, enabled: boolean, phoneNumber?: string) {
  return apiRequest<{ success: boolean; clone: ApiClone }>(`/api/clones/${cloneId}/whatsapp`, {
    method: "PATCH",
    body: { enabled, phone_number: phoneNumber ?? "" },
  });
}

export async function deleteClone(cloneId: string) {
  return apiRequest<{ success: boolean }>(`/api/clones/${cloneId}`, {
    method: "DELETE",
  });
}

export async function chatWithClone(cloneId: string, message: string) {
  return apiRequest<{ response: string; documents: Array<{ filename?: string; unique_filename?: string; size?: number; description?: string }>; clone_id: string }>(
    `/api/chat/${cloneId}`,
    {
      method: "POST",
      body: { message },
    }
  );
}

export async function clearCloneChat(cloneId: string) {
  return apiRequest<{ success: boolean; message: string }>(`/api/clear_chat/${cloneId}`, {
    method: "POST",
  });
}

export async function getCloneChunks(cloneId: string) {
  return apiRequest<{ chunks: ApiChunk[]; count: number }>(`/api/chunks/${cloneId}`);
}

export async function previewTextChunks(cloneId: string, textContent: string) {
  const formData = new FormData();
  formData.append("text_content", textContent);
  return apiFormRequest<{ success: boolean; chunks: ApiChunkPreview[]; total_chunks: number }>(`/preview_chunks/${cloneId}`, formData);
}

export async function previewFileChunks(cloneId: string, file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return apiFormRequest<{ success: boolean; chunks: ApiChunkPreview[]; total_chunks: number }>(`/preview_chunks/${cloneId}`, formData);
}

export async function uploadSelectedChunks(cloneId: string, selectedIndices: number[]) {
  return apiRequest<{ success: boolean; chunks_added: number; total_chunks: number }>(`/upload_selected_chunks/${cloneId}`, {
    method: "POST",
    body: { selected_indices: selectedIndices },
  });
}

export async function uploadTextChunks(cloneId: string, textContent: string) {
  const formData = new FormData();
  formData.append("text_content", textContent);
  return apiFormRequest<{ success: boolean; chunks_added: number; total_chunks: number }>(`/upload_data/${cloneId}`, formData);
}

export async function uploadFileChunks(cloneId: string, file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return apiFormRequest<{ success: boolean; chunks_added: number; total_chunks: number }>(`/upload_data/${cloneId}`, formData);
}

export async function deleteChunks(cloneId: string, chunkIds: string[]) {
  return apiRequest<{ success: boolean; deleted_count: number; remaining_chunks: number }>(`/delete_chunks/${cloneId}`, {
    method: "POST",
    body: { chunk_ids: chunkIds },
  });
}

export async function uploadDocument(cloneId: string, description: string, file: File) {
  const formData = new FormData();
  formData.append("description", description);
  formData.append("document", file);
  return apiFormRequest<{ success: boolean; message: string; filename: string; total_chunks: number }>(`/upload_document/${cloneId}`, formData);
}

export async function getCloneDocuments(cloneId: string) {
  return apiRequest<{ documents: ApiDocument[] }>(`/api/documents/${cloneId}`);
}
