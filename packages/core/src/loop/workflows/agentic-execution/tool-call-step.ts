import type { ToolSet } from 'ai-v5';
import { zodToJsonSchema } from 'zod-to-json-schema';
import type { JSONSchema7 } from 'json-schema';
import type { OutputSchema } from '../../../stream/base/schema';
import { ChunkFrom } from '../../../stream/types';
import type { MastraToolInvocationOptions } from '../../../tools/types';
import { createStep } from '../../../workflows';
import type { OuterLLMRun } from '../../types';
import { toolCallInputSchema, toolCallOutputSchema } from '../schema';

/**
 * Error thrown when a tool is not found
 */
class NoSuchToolError extends Error {
  readonly toolName: string;

  constructor(toolName: string) {
    super(`Tool "${toolName}" not found`);
    this.name = 'NoSuchToolError';
    this.toolName = toolName;
  }
}

/**
 * Error thrown when tool input validation fails
 */
class InvalidToolInputError extends Error {
  readonly toolName: string;
  readonly input: unknown;

  constructor(toolName: string, input: unknown, message: string) {
    super(`Invalid input for tool "${toolName}": ${message}`);
    this.name = 'InvalidToolInputError';
    this.toolName = toolName;
    this.input = input;
  }
}

/**
 * Get JSON Schema for a tool's input
 */
function getToolInputSchema(tool: any): JSONSchema7 | undefined {
  if (!tool) return undefined;

  // Check for inputSchema (Mastra tools)
  if (tool.inputSchema) {
    try {
      return zodToJsonSchema(tool.inputSchema) as JSONSchema7;
    } catch {
      return undefined;
    }
  }

  // Check for parameters (Vercel AI SDK tools)
  if (tool.parameters) {
    try {
      return zodToJsonSchema(tool.parameters) as JSONSchema7;
    } catch {
      return undefined;
    }
  }

  return undefined;
}

export function createToolCallStep<
  Tools extends ToolSet = ToolSet,
  OUTPUT extends OutputSchema | undefined = undefined,
>({
  tools,
  messageList,
  options,
  writer,
  controller,
  runId,
  streamState,
  modelSpanTracker,
  repairToolCall,
}: OuterLLMRun<Tools, OUTPUT>) {
  return createStep({
    id: 'toolCallStep',
    inputSchema: toolCallInputSchema,
    outputSchema: toolCallOutputSchema,
    execute: async ({ inputData, suspend, resumeData, requestContext }) => {
      // If the tool was already executed by the provider, skip execution
      if (inputData.providerExecuted) {
        return {
          ...inputData,
          result: inputData.output,
        };
      }

      let toolName = inputData.toolName;
      let args = inputData.args;
      let toolCallId = inputData.toolCallId;

      let tool =
        tools?.[toolName] ||
        Object.values(tools || {})?.find(t => `id` in t && t.id === toolName);

      // If tool not found and repairToolCall is provided, try to repair
      if (!tool && repairToolCall) {
        const error = new NoSuchToolError(toolName);

        const repaired = await repairToolCall({
          toolCall: {
            toolCallId,
            toolName,
            input: typeof args === 'string' ? args : JSON.stringify(args),
          },
          tools: tools as Tools,
          inputSchema: ({ toolName: tn }) => {
            const t = tools?.[tn] || Object.values(tools || {})?.find(x => `id` in x && x.id === tn);
            return getToolInputSchema(t);
          },
          error,
        });

        if (repaired) {
          // Update with repaired values
          toolCallId = repaired.toolCallId;
          toolName = repaired.toolName;
          try {
            args = JSON.parse(repaired.input);
          } catch {
            // If input is not valid JSON, keep original args
          }

          // Try to find the tool again with the repaired name
          tool = tools?.[toolName] || Object.values(tools || {})?.find(t => `id` in t && t.id === toolName);
        }
      }

      if (!tool) {
        throw new NoSuchToolError(toolName);
      }

      if (tool && 'onInputAvailable' in tool) {
        try {
          await tool?.onInputAvailable?.({
            toolCallId,
            input: args,
            messages: messageList.get.input.aiV5.model(),
            abortSignal: options?.abortSignal,
          });
        } catch (error) {
          console.error('Error calling onInputAvailable', error);
        }
      }

      if (!tool.execute) {
        return { ...inputData, toolName, toolCallId, args };
      }

      try {
        const requireToolApproval = requestContext.get('__mastra_requireToolApproval');
        if (requireToolApproval || (tool as any).requireApproval) {
          if (!resumeData) {
            controller.enqueue({
              type: 'tool-call-approval',
              runId,
              from: ChunkFrom.AGENT,
              payload: {
                toolCallId,
                toolName,
                args,
              },
            });
            return suspend(
              {
                requireToolApproval: {
                  toolCallId,
                  toolName,
                  args,
                },
                __streamState: streamState.serialize(),
              },
              {
                resumeLabel: toolCallId,
              },
            );
          } else {
            if (!resumeData.approved) {
              return {
                result: 'Tool call was not approved by the user',
                ...inputData,
                toolName,
                toolCallId,
                args,
              };
            }
          }
        }

        const toolOptions: MastraToolInvocationOptions = {
          abortSignal: options?.abortSignal,
          toolCallId,
          messages: messageList.get.input.aiV5.model(),
          writableStream: writer,
          // Pass current step span as parent for tool call spans
          tracingContext: modelSpanTracker?.getTracingContext(),
          suspend: async (suspendPayload: any) => {
            controller.enqueue({
              type: 'tool-call-suspended',
              runId,
              from: ChunkFrom.AGENT,
              payload: { toolCallId, toolName, suspendPayload },
            });

            return await suspend(
              {
                toolCallSuspended: suspendPayload,
                __streamState: streamState.serialize(),
              },
              {
                resumeLabel: toolCallId,
              },
            );
          },
          resumeData,
        };

        const result = await tool.execute(args, toolOptions);
        return { result, ...inputData, toolName, toolCallId, args };
      } catch (error) {
        return {
          error: error as Error,
          ...inputData,
          toolName,
          toolCallId,
          args,
        };
      }
    },
  });
}
