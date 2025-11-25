---
"@mastra/core": minor
---

Add `repairToolCall` option to `agent.stream` for handling invalid tool calls

- Added `ToolCallRepairContext`, `RepairedToolCall`, and `RepairToolCallFunction` types
- Added `repairToolCall` option to `AgentExecutionOptions` and `AgentStreamOptions`
- Implemented repair logic in tool call execution step

