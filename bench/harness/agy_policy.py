"""Native PreToolUse hook: keep this run within the benchmark's local coding tools."""
import json
from pathlib import Path
import sys
import time

event=json.load(sys.stdin)
result={'decision':'deny','reason':'This benchmark permits local code, shell and validation tools only. External research, browser, plugins and delegation are disabled.'}
log=Path.home()/'.gemini/antigravity-cli/tool_policy.jsonl'
with log.open('a') as out:
    out.write(json.dumps({'ts':time.time(),'tool':event.get('toolCall',{}).get('name'),
        'conversation_id':event.get('conversationId'),**result})+'\n')
print(json.dumps(result))
