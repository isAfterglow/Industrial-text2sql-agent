import { useEffect, useMemo, useState } from "react";
import {
  Alert, Button, Drawer, Empty, Input, Layout, List, Segmented, Select, Space,
  Spin, Statistic, Table, Tabs, Tag, Timeline, Typography, message,
} from "antd";
import {
  CheckCircleOutlined, ClockCircleOutlined, DatabaseOutlined, PlayCircleOutlined,
  SafetyCertificateOutlined, SendOutlined, ThunderboltOutlined,
} from "@ant-design/icons";
import { api, type Approval, type Profile, type Task } from "./api";

const { Header, Sider, Content } = Layout;
const { TextArea } = Input;
const { Text, Title } = Typography;

type NodeEvent = { sequence: number; created_at: string; payload: { type: string; event?: { node: string; status: string; elapsed_ms: number; output?: Record<string, unknown> } } };

const statusColor: Record<string, string> = {
  queued: "default", running: "processing", completed: "success", approval_required: "warning", failed: "error",
};

function formatMs(value: unknown) {
  const number = Number(value || 0);
  return number >= 1000 ? `${(number / 1000).toFixed(2)} s` : `${number.toFixed(0)} ms`;
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="code-block">{JSON.stringify(value, null, 2)}</pre>;
}

function TaskResult({ task }: { task: Task }) {
  const result = task.result || {};
  const rows = Array.isArray(result.rows) ? result.rows : [];
  const columns = Array.isArray(result.columns) ? result.columns : [];
  const tableColumns = columns.map((name, index) => ({
    title: String(name), dataIndex: String(index), key: String(index),
    render: (value: unknown) => value === null ? <Text type="secondary">null</Text> : String(value),
  }));
  const data = rows.map((row, index) => {
    const item: Record<string, unknown> = { key: index };
    if (Array.isArray(row)) row.forEach((value, column) => { item[String(column)] = value; });
    return item;
  });
  const modelCalls = Array.isArray(result.model_calls) ? result.model_calls : [];

  return <div className="result-area">
    <div className="result-heading">
      <div>
        <Text type="secondary">FINAL RESPONSE</Text>
        <Title level={4}>{String(result.final_answer || task.error_message || "Waiting for the agent to finish")}</Title>
      </div>
      <Space size={18} className="metrics">
        <Statistic title="Rows" value={Number(result.row_count || 0)} />
        <Statistic title="Retries" value={Number(result.retry_count || 0)} />
        <Statistic title="Elapsed" value={formatMs(result.elapsed_ms)} />
      </Space>
    </div>
    {task.status === "failed" && <Alert type="error" showIcon message={task.error_message} />}
    {task.status === "approval_required" && <Alert type="warning" showIcon message="Execution paused for human approval." />}
    <Tabs className="result-tabs" items={[
      {
        key: "data", label: "Data", children: data.length
          ? <Table size="small" pagination={{ pageSize: 8, hideOnSinglePage: true }} scroll={{ x: true }} columns={tableColumns} dataSource={data} />
          : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No result rows" />,
      },
      { key: "sql", label: "SQL", children: <pre className="sql-block">{String(result.sql || "SQL will appear after planning.")}</pre> },
      { key: "plan", label: "Plan", children: <JsonBlock value={result.advanced_plan || result.query_spec || {}} /> },
      {
        key: "models", label: `Models (${modelCalls.length})`, children:
          <JsonBlock value={modelCalls} />,
      },
    ]} />
  </div>;
}

function TracePanel({ events, task }: { events: NodeEvent[]; task: Task | null }) {
  const nodes = events.filter((item) => item.payload.type === "node" && item.payload.event).map((item) => item.payload.event!);
  const result = task?.result || {};
  const fewShot = result.few_shot as Record<string, unknown> | undefined;
  return <aside className="trace-panel">
    <div className="panel-title"><ThunderboltOutlined /> Execution trace</div>
    {task && <div className="trace-meta">
      <Tag color={statusColor[task.status]}>{task.status}</Tag>
      <Text type="secondary">{task.trace_id}</Text>
    </div>}
    {nodes.length ? <Timeline items={nodes.map((event) => ({
      color: event.status === "failed" ? "red" : event.status === "rejected" ? "orange" : "green",
      children: <div className="trace-node"><b>{event.node}</b><span>{formatMs(event.elapsed_ms)}</span></div>,
    }))} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Node events stream here" />}
    <div className="trace-section">
      <Text type="secondary">MEMORY RETRIEVAL</Text>
      <JsonBlock value={fewShot || { status: "No completed task selected" }} />
    </div>
    <div className="trace-section">
      <Text type="secondary">FAILURE EVENTS</Text>
      <JsonBlock value={result.failure_events || []} />
    </div>
  </aside>;
}

function ApprovalDrawer({ approval, profile, onClose, onResumed }: {
  approval: Approval | null; profile: Profile; onClose: () => void; onResumed: (task: Task) => void;
}) {
  const [actor, setActor] = useState("reviewer");
  const [comment, setComment] = useState("");
  const [plan, setPlan] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => setPlan(JSON.stringify(approval?.payload.advanced_plan || {}, null, 2)), [approval]);

  async function decide(action: "approved" | "rejected" | "edited_plan") {
    if (!approval) return;
    setBusy(true);
    try {
      const body: Record<string, unknown> = { action, actor, comment };
      if (action === "edited_plan") body.advanced_plan = JSON.parse(plan);
      await api.decideApproval(approval.approval_id, profile, body);
      message.success(`Approval recorded: ${action}`);
      if (action !== "rejected") {
        const task = await api.resumeApproval(approval.approval_id, profile);
        onResumed(task);
      }
      onClose();
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Approval request failed");
    } finally { setBusy(false); }
  }

  return <Drawer title="Human approval" width={580} open={Boolean(approval)} onClose={onClose}>
    {approval && <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert type="warning" showIcon message={String((approval.payload.risk as Record<string, unknown> | undefined)?.reasons?.toString() || "Execution risk requires review")} />
      <div><Text type="secondary">QUESTION</Text><p>{String(approval.payload.question || "")}</p></div>
      <div><Text type="secondary">COMPILED SQL</Text><pre className="sql-block">{String(approval.payload.compiled_sql || "")}</pre></div>
      <div><Text type="secondary">REVIEWER</Text><Input value={actor} onChange={(event) => setActor(event.target.value)} /></div>
      <div><Text type="secondary">COMMENT</Text><TextArea value={comment} onChange={(event) => setComment(event.target.value)} rows={2} /></div>
      <div><Text type="secondary">ADVANCED PLAN (only for an edited plan)</Text><TextArea className="plan-editor" value={plan} onChange={(event) => setPlan(event.target.value)} rows={10} /></div>
      <Space wrap>
        <Button type="primary" icon={<CheckCircleOutlined />} loading={busy} onClick={() => decide("approved")}>Approve and run</Button>
        <Button icon={<SafetyCertificateOutlined />} loading={busy} onClick={() => decide("edited_plan")}>Save edited plan</Button>
        <Button danger loading={busy} onClick={() => decide("rejected")}>Reject</Button>
      </Space>
    </Space>}
  </Drawer>;
}

export function App() {
  const [profile, setProfile] = useState<Profile>("resin");
  const [question, setQuestion] = useState("Find the top 3 samples by original density.");
  const [forceApproval, setForceApproval] = useState(false);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [events, setEvents] = useState<NodeEvent[]>([]);
  const [approval, setApproval] = useState<Approval | null>(null);
  const [pendingApproval, setPendingApproval] = useState<Approval | null>(null);
  const [memories, setMemories] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(false);
  const selected = tasks.find((task) => task.task_id === selectedId) || null;

  async function refreshTasks() {
    try { setTasks(await api.listTasks()); } catch { /* API may be starting */ }
  }
  async function refreshGovernance() {
    try {
      const [items, records] = await Promise.all([api.listApprovals(profile, "pending"), api.listMemories(profile)]);
      setMemories(records);
      setPendingApproval(items[0] || null);
    } catch { /* task execution remains available without this panel */ }
  }
  useEffect(() => { void refreshTasks(); const id = window.setInterval(() => void refreshTasks(), 1800); return () => window.clearInterval(id); }, []);
  useEffect(() => { void refreshGovernance(); }, [profile]);
  useEffect(() => {
    if (!selectedId) return;
    setEvents([]);
    const source = new EventSource(`/api/tasks/${selectedId}/events`);
    source.addEventListener("node", (raw) => setEvents((previous) => [...previous, JSON.parse((raw as MessageEvent).data)]));
    source.addEventListener("terminal", () => { source.close(); void refreshTasks(); void refreshGovernance(); });
    source.onerror = () => source.close();
    return () => source.close();
  }, [selectedId]);

  async function submit() {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const task = await api.createTask({ question, profile, force_approval: forceApproval });
      setTasks((previous) => [task, ...previous]);
      setSelectedId(task.task_id);
    } catch (error) { message.error(error instanceof Error ? error.message : "Could not start task"); }
    finally { setLoading(false); }
  }

  const memoryCounts = useMemo(() => memories.reduce<Record<string, number>>((counts, item) => {
    const key = String(item.memory_type || "unknown"); counts[key] = (counts[key] || 0) + 1; return counts;
  }, {}), [memories]);

  return <Layout className="app-shell">
    <Header className="topbar">
      <div className="brand"><DatabaseOutlined /><span>Text2SQL Agent</span><Tag>WORKBENCH</Tag></div>
      <div className="topbar-status"><span className="live-dot" /> API + SSE</div>
    </Header>
    <Layout>
      <Sider width={274} className="left-nav">
        <div className="nav-section">
          <Text type="secondary">RECENT TASKS</Text>
          <List dataSource={tasks.slice(0, 14)} locale={{ emptyText: "No tasks yet" }} renderItem={(task) =>
            <List.Item className={task.task_id === selectedId ? "task-item selected" : "task-item"} onClick={() => setSelectedId(task.task_id)}>
              <div className="task-label"><b>{task.profile === "resin" ? "Resin" : "Steel"}</b><Tag color={statusColor[task.status]}>{task.status}</Tag></div>
              <Text ellipsis>{task.question}</Text>
            </List.Item>} />
        </div>
        <div className="nav-section memory-summary">
          <Text type="secondary">MEMORY GOVERNANCE</Text>
          <div><span>Formal</span><b>{memoryCounts.episodic || 0}</b></div>
          <div><span>Candidate</span><b>{memoryCounts.candidate_episodic || 0}</b></div>
          <div><span>Semantic</span><b>{memoryCounts.semantic || 0}</b></div>
        </div>
      </Sider>
      <Content className="main-content">
        <section className="query-bar">
          <div className="query-controls">
            <Select value={profile} onChange={setProfile} options={[{ value: "resin", label: "Resin materials" }, { value: "steel_industry", label: "Steel industry" }]} />
            <Segmented value={forceApproval ? "approval" : "standard"} onChange={(value) => setForceApproval(value === "approval")} options={[{ label: "Standard", value: "standard" }, { label: "Force approval", value: "approval" }]} />
          </div>
          <TextArea value={question} onChange={(event) => setQuestion(event.target.value)} autoSize={{ minRows: 2, maxRows: 5 }} onPressEnter={(event) => { if (!event.shiftKey) { event.preventDefault(); void submit(); } }} />
          <Button type="primary" icon={<SendOutlined />} loading={loading} onClick={() => void submit()}>Run query</Button>
        </section>
        <section className="workspace">
          <main className="result-panel">{selected ? <TaskResult task={selected} /> : <Empty description="Submit a query to start an agent task" />}</main>
          <TracePanel events={events} task={selected} />
        </section>
        <section className="approval-strip">
          <Space><SafetyCertificateOutlined /><Text>Pending approvals are governed by immutable plan snapshots.</Text></Space>
          <Button size="small" icon={<ClockCircleOutlined />} onClick={() => void refreshGovernance()}>Refresh governance</Button>
          {pendingApproval && <Button size="small" type="primary" icon={<PlayCircleOutlined />} onClick={() => setApproval(pendingApproval)}>Review pending approval</Button>}
        </section>
      </Content>
    </Layout>
    <ApprovalDrawer approval={approval} profile={profile} onClose={() => setApproval(null)} onResumed={(task) => { setTasks((previous) => [task, ...previous]); setSelectedId(task.task_id); }} />
  </Layout>;
}
