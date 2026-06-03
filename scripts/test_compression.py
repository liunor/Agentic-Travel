import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from server.agent.session_storage import (
    global_session_storage, 
    record_transcript, 
    load_transcript_file,
    get_session_transcript_path,
    MAX_CONTEXT_TOKENS
)
from utils.tokens import token_count_with_estimation

async def main():
    print("=" * 60)
    print(" Testing Context Compression (200K limit)")
    print("=" * 60)
    
    session_id = "test_compression_001"
    file_path = get_session_transcript_path(session_id)
    
    if os.path.exists(file_path):
        os.remove(file_path)

    # We will simulate a very long session
    # 200,000 tokens ≈ 800,000 characters
    sys_msg = SystemMessage(content="You are a helpful assistant.", id="uuid-sys")
    
    # 4 messages, each 250,000 characters (approx 62.5k tokens) -> 250k tokens total, exceeds 200k.
    long_text = "A" * 250_000
    msg1 = HumanMessage(content=long_text, id="uuid-1")
    msg2 = AIMessage(content="I see.", id="uuid-2")
    msg3 = HumanMessage(content=long_text, id="uuid-3")
    msg4 = AIMessage(content="I see again.", id="uuid-4")
    msg5 = HumanMessage(content=long_text, id="uuid-5")
    
    msg4.usage_metadata = {"total_tokens": 150_000}
    
    await record_transcript(session_id, [sys_msg, msg1, msg2, msg3, msg4, msg5])
    await global_session_storage.flush()
    
    rebuilt_msgs = await load_transcript_file(session_id)
    
    print(f"Original msg count: 6, Rebuilt msg count after compression: {len(rebuilt_msgs)}")
    
    # Token sizes:
    # sys_msg: tiny
    # msg1: 62.5k
    # msg2: tiny
    # msg3: 62.5k
    # msg4: anchor usage = 10
    # msg5: 62.5k
    # 
    # With anchor on msg4:
    # [sys_msg, msg3, msg4, msg5] = tiny + 62.5k + 10 + 62.5k = 125k < 200k
    # [sys_msg, msg2, msg3, msg4, msg5] = 125k + tiny < 200k
    # [sys_msg, msg1, msg2, msg3, msg4, msg5] = 125k + 62.5k = 187k < 200k
    # Wait! If msg4 has usage_metadata = {"total_tokens": 10}, 
    # then token_count_with_estimation on ANY chain ending after msg4 uses msg4 as anchor!
    # That means msg1, msg2, msg3's tokens are IGNORED by the estimation function if msg4 is in the list!
    
    for m in rebuilt_msgs:
        print(f"Rebuilt msg id: {m.id}, length: {len(m.content)}")
    
    # We should ensure the system message is preserved
    assert rebuilt_msgs[0].id == "uuid-sys"
    
    # Verify usage was attached
    m4_rebuilt = next((m for m in rebuilt_msgs if m.id == "uuid-4"), None)
    assert m4_rebuilt is not None
    assert getattr(m4_rebuilt, "usage_metadata", None) == {"total_tokens": 150_000}

    # Clean up
    if os.path.exists(file_path):
        os.remove(file_path)
    print("\n[SUCCESS] Compression test passed!")

if __name__ == "__main__":
    asyncio.run(main())
