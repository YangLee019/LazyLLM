from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, loop, locals, package, FileSystemQueue, once_wrapper
from .toolsManager import ToolManager
from typing import List, Any, Dict, Union, Callable, Optional
from .base import LazyLLMAgentBase, _write_agent_data
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER
from lazyllm.common.deprecated import deprecated
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase, create_sandbox
import re
import json

FC_PROMPT = f'''# Tools

## You have access to the following tools:
## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs.
{FC_PROMPT_PLACEHOLDER}

Don\'t make assumptions about what values to plug into functions.
Ask for clarification if a user request is ambiguous.\n
'''


class StreamResponse():
    def __init__(self, prefix: str, prefix_color: str = None, color: str = None, stream: bool = False):
        self.stream = stream
        self.prefix = prefix
        self.prefix_color = prefix_color
        self.color = color

    def __call__(self, *inputs):
        if self.stream: FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': f'\n{self.prefix}\n'}))
        if len(inputs) == 1:
            if self.stream: FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': f'{inputs[0]}'}))
            return inputs[0]
        if self.stream: FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': f'{inputs}'}))
        return package(*inputs)


_COMPACTION_TRUNCATE_LEN = 200
_CURRENT_TOOL_RESULT_TRUNCATE_LEN = 3000
_ASSISTANT_CONTENT_TRUNCATE_LEN = 800
_ASSISTANT_REASONING_TRUNCATE_LEN = 600
_MAX_COMPACT_COLLECTION_ITEMS = 12


def _truncate_text(text: Any, limit: int) -> str:
    content = '' if text is None else str(text)
    if limit <= 0 or len(content) <= limit:
        return content
    return f'[truncated {len(content)} chars] {content[:limit]}...'


def _compact_tool_result_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        limit = _CURRENT_TOOL_RESULT_TRUNCATE_LEN if depth == 0 else max(256, _CURRENT_TOOL_RESULT_TRUNCATE_LEN // 4)
        return _truncate_text(value, limit)
    if isinstance(value, list):
        kept = [_compact_tool_result_value(item, depth=depth + 1) for item in value[:_MAX_COMPACT_COLLECTION_ITEMS]]
        if len(value) > _MAX_COMPACT_COLLECTION_ITEMS:
            kept.append(f'...[truncated {len(value) - _MAX_COMPACT_COLLECTION_ITEMS} items]...')
        return kept
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        items = list(value.items())
        for key, item_value in items[:_MAX_COMPACT_COLLECTION_ITEMS]:
            compacted[str(key)] = _compact_tool_result_value(item_value, depth=depth + 1)
        if len(items) > _MAX_COMPACT_COLLECTION_ITEMS:
            compacted['__truncated_keys__'] = len(items) - _MAX_COMPACT_COLLECTION_ITEMS
        return compacted
    return value


def _serialize_tool_result(value: Any, *, limit: int = _CURRENT_TOOL_RESULT_TRUNCATE_LEN) -> str:
    compacted = _compact_tool_result_value(value)
    if isinstance(compacted, str):
        return _truncate_text(compacted, limit)
    try:
        serialized = json.dumps(compacted, ensure_ascii=False)
    except TypeError:
        serialized = str(compacted)
    return _truncate_text(serialized, limit)


def _compact_assistant_message(message: Dict[str, Any]) -> Dict[str, Any]:
    compacted = dict(message)
    if 'content' in compacted:
        compacted['content'] = _truncate_text(compacted.get('content', ''), _ASSISTANT_CONTENT_TRUNCATE_LEN)
    reasoning = compacted.get('reasoning_content')
    if reasoning:
        compacted['reasoning_content'] = _truncate_text(reasoning, _ASSISTANT_REASONING_TRUNCATE_LEN)
    return compacted


def _compact_chat_history(history: List[Dict[str, Any]], keep_full_turns: int) -> List[Dict[str, Any]]:
    tool_indices = [i for i, m in enumerate(history) if m.get('role') == 'tool']
    assistant_tool_turn_indices = [
        i for i, m in enumerate(history)
        if m.get('role') == 'assistant' and isinstance(m.get('tool_calls'), list) and m.get('tool_calls')
    ]
    tool_cutoff = len(tool_indices) - keep_full_turns
    assistant_cutoff = len(assistant_tool_turn_indices) - keep_full_turns
    to_truncate = set(tool_indices[:tool_cutoff]) if tool_cutoff > 0 else set()
    to_compact_assistants = (
        set(assistant_tool_turn_indices[:assistant_cutoff]) if assistant_cutoff > 0 else set()
    )
    if not to_truncate and not to_compact_assistants:
        return list(history)
    result = []
    for i, msg in enumerate(history):
        if i in to_compact_assistants:
            msg = _compact_assistant_message(msg)
        if i in to_truncate:
            msg = dict(msg, content=_serialize_tool_result(msg.get('content', ''), limit=_COMPACTION_TRUNCATE_LEN))
        result.append(msg)
    return result


class FunctionCall(ModuleBase):

    def __init__(self, llm, tools: Optional[List[Union[str, Callable]]] = None, *, return_trace: bool = False,
                 stream: bool = False, _prompt: str = None, _tool_manager: Optional[ToolManager] = None,
                 skill_manager=None, sandbox: Optional[LazyLLMSandboxBase] = None,
                 keep_full_turns: int = 0):
        super().__init__(return_trace=return_trace)
        if _tool_manager is None:
            assert tools, 'tools cannot be empty.'
            self._sandbox = sandbox or create_sandbox()
            self._tools_manager = ToolManager(tools, return_trace=return_trace, sandbox=self._sandbox)
        else:
            self._tools_manager = _tool_manager
            self._sandbox = _tool_manager.sandbox
        self._skill_manager = skill_manager
        self._stream = stream
        self._keep_full_turns = keep_full_turns
        prompt = _prompt or FC_PROMPT
        self._prompter = ChatPrompter(
            instruction={'system': prompt, 'user': ''},
            tools=lambda: self._tools_manager.tools_description,
            skills=self._skill_manager.build_prompt() if self._skill_manager else '',
        )
        self._llm = llm.share(
            prompt=self._prompter,
            format=FunctionCallFormatter(),
            stream=stream,
        ).used_by(self._module_id)
        with pipeline() as self._impl:
            self._impl.pre_action = self._build_history
            self._impl.llm = self._llm
            self._impl.post_action = self._post_action

    @property
    def sandbox(self) -> LazyLLMSandboxBase:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox: Optional[LazyLLMSandboxBase]):
        self._sandbox = sandbox
        if hasattr(self, '_tools_manager') and self._tools_manager is not None:
            self._tools_manager.sandbox = sandbox

    def _build_history(self, input: Union[str, dict, list]):
        workspace = locals['_lazyllm_agent']['workspace']
        history_idx = len(workspace.setdefault('history', []))
        if isinstance(input, str):
            workspace['history'].append({'role': 'user', 'content': input})
        elif isinstance(input, dict) and 'input' in input:
            workspace['history'].append(
                {'role': 'user', 'content': input.get('input', '')}
            )
        elif isinstance(input, dict) and input.get('role') == 'user':
            workspace['history'].append(
                {'role': 'user', 'content': input.get('content', '')}
            )
        elif isinstance(input, dict):
            tool_call_results = [
                {
                    'role': 'tool',
                    'content': _serialize_tool_result(tool_call.get('tool_call_result')),
                    'tool_call_id': tool_call['id'],
                    'name': tool_call['function']['name'],
                } for tool_call in workspace['tool_call_trace']
            ]
            workspace['history'].append(_compact_assistant_message({
                'role': 'assistant',
                'content': input.get('content', ''),
                'tool_calls': input.get('tool_calls', []),
                'reasoning_content': input.get('reasoning_content', ''),
            }))
            input = {'input': tool_call_results}
            history_idx += 1
            workspace['history'].extend(tool_call_results)
        chat_history = workspace['history'][:history_idx]
        if self._keep_full_turns > 0:
            chat_history = _compact_chat_history(chat_history, self._keep_full_turns)
        locals['chat_history'][self._llm._module_id] = chat_history
        return input

    def _post_action(self, llm_output: Dict[str, Any]):
        if not llm_output.get('tool_calls'):
            if (match := re.search(r'Action:\s*Call\s+(\w+)\s+with\s+parameters\s+(\{.*?\})', llm_output['content'])):
                try:
                    llm_output['tool_calls'] = [{'function': {'name': match.group(1),
                                                              'arguments': json.loads(match.group(2))}}]
                except Exception: pass
        if tool_calls := llm_output.get('tool_calls'):
            if isinstance(tool_calls, list): [item.pop('index', None) for item in tool_calls]
            if self._stream:
                _write_agent_data('tool_calls', tool_calls=tool_calls)
            tool_calls_results = self._tools_manager(tool_calls)
            if self._stream:
                _write_agent_data('tool_results',
                                  tool_results=LazyLLMAgentBase._normalize_tool_results(tool_calls,
                                                                                        tool_calls_results))
            locals['_lazyllm_agent']['workspace']['tool_call_trace'] = [
                {**tool_call, 'tool_call_result': tool_result}
                for tool_call, tool_result in zip(tool_calls, tool_calls_results)
            ]
        else:
            llm_output = llm_output['content']
        return llm_output

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        if 'workspace' not in locals['_lazyllm_agent']:
            locals['_lazyllm_agent']['workspace'] = dict(history=llm_chat_history or [])
        result = self._impl(input)

        # If the model decides not to call any tools, the result is a string. For debugging and subsequent tasks,
        # the last non-empty tool call trace is stored in locals['_lazyllm_agent']['completed']
        # and history is stored in locals['_lazyllm_agent']['history'].
        if isinstance(result, str):
            workspace = locals['_lazyllm_agent'].pop('workspace', {})
            locals['_lazyllm_agent']['completed'] = workspace.pop(
                'tool_call_trace', locals['_lazyllm_agent'].get('completed', []))
            locals['_lazyllm_agent']['history'] = workspace.pop('history', [])
            locals['chat_history'][self._llm._module_id] = []
        return result

@deprecated('ReactAgent')
class FunctionCallAgent(LazyLLMAgentBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False, stream: bool = False,
                 return_last_tool_calls: bool = False,
                 skills: Union[bool, str, List[str], None] = None, desc: str = '',
                 workspace: Optional[str] = None, fs: Optional[Any] = None,
                 skills_dir: Optional[str] = None, enable_builtin_tools: bool = True):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries,
                         return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls,
                         skills=skills, desc=desc, workspace=workspace, fs=fs, skills_dir=skills_dir,
                         enable_builtin_tools=enable_builtin_tools)
        assert self._llm is not None, 'llm cannot be empty.'
        self._assert_tools()
        prompt = self._append_workspace_prompt(FC_PROMPT)
        self._fc = FunctionCall(llm=self._llm, return_trace=return_trace, stream=stream,
                                _prompt=prompt, _tool_manager=self._tools_manager,
                                skill_manager=self._skill_manager)
        self._fc._llm.used_by(self._module_id)

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        agent = loop(self._fc, stop_condition=lambda x: isinstance(x, str), count=self._max_retries)
        self._agent = agent

    def _pre_process(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        if llm_chat_history is not None:
            return (query, llm_chat_history)
        return query

    def _post_process(self, ret):
        if isinstance(ret, str):
            completed = self._pop_tool_calls()
            if completed is not None:
                return completed
            return ret
        raise ValueError(f'After retrying {self._max_retries} times, the function call agent still fails to call '
                         f'successfully.')
