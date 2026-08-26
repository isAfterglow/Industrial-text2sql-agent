export type Profile = "resin" | "steel_industry";

export type Task = {
  task_id: string;
  status: "queued" | "running" | "completed" | "approval_required" | "failed";
  profile: Profile;
  question: string;
  session_id: string;
  trace_id: string;
  approval_id: string;
  result: Record<string, unknown>;
  error_message: string;
  created_at: string;
  started_at: string;
  finished_at: string;
};

export type Approval = {
  approval_id: string;
  profile: Profile;
  status: string;
  payload: Record<string, unknown>;
  decision: Record<string, unknown>;
  created_at: string;
};

export type ApprovalAudit = {
  audit_id: string;
  actor_id: string;
  action: string;
  comment: string;
  payload: Record<string, unknown>;
  created_at: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `Request failed: ${response.status}`);
  return response.json() as Promise<T>;
}

export const api = {
  createTask: (body: { question: string; profile: Profile; session_id?: string; force_approval?: boolean }) =>
    request<Task>("/api/tasks", { method: "POST", body: JSON.stringify(body) }),
  getTask: (id: string) => request<Task>(`/api/tasks/${id}`),
  listTasks: () => request<Task[]>("/api/tasks"),
  listApprovals: (profile: Profile, status?: string) => request<Approval[]>(`/api/approvals?profile=${profile}${status ? `&status=${status}` : ""}`),
  decideApproval: (id: string, profile: Profile, body: Record<string, unknown>) =>
    request<Approval>(`/api/approvals/${id}/decision?profile=${profile}`, { method: "POST", body: JSON.stringify(body) }),
  resumeApproval: (id: string, profile: Profile) =>
    request<Task>(`/api/approvals/${id}/resume?profile=${profile}`, { method: "POST" }),
  approvalAudit: (id: string, profile: Profile) =>
    request<ApprovalAudit[]>(`/api/approvals/${id}/audit?profile=${profile}`),
  listMemories: (profile: Profile) => request<Record<string, unknown>[]>(`/api/memories?profile=${profile}`),
};
