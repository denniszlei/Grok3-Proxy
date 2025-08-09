import asyncio
import json
import base64
import hashlib
import secp256k1
import re
import uuid
import time
from typing import Dict, List, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import aiohttp
from curl_cffi.requests import AsyncSession

# === Configuration ===
@dataclass
class Config:
    STATSIG_POOL_SIZE = 5
    STATSIG_REFRESH_THRESHOLD = 2
    MAX_CONCURRENT_REQUESTS = 100
    REQUEST_TIMEOUT = 30
    BASE_URL = "https://grok.com"
    BUNDLE_URL = "https://grok.com/_next/static/chunks/24823-4b48d1b762350b41.js"
    API_CREATE_ANON = "https://grok.com/rest/auth/create-anon-user"
    API_CHALLENGE = "https://grok.com/rest/auth/create-anon-user-challenge"

config = Config()

# === Models ===
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "grok-3"
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

# === Global State ===
class ServerState:
    def __init__(self):
        self.statsig_pool: asyncio.Queue = asyncio.Queue()
        self.session_pool: List[AsyncSession] = []
        self.bundle_cache: Optional[str] = None
        self.set_anon_id_cache: Optional[str] = None
        self.harvesting_lock = asyncio.Lock()
        self.request_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

state = ServerState()

# === Auth Token Generation ===
async def create_http_session() -> AsyncSession:
    """Create configured HTTP session"""
    session = AsyncSession(impersonate="chrome120")
    session.headers.update({
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "accept-language": "en-US,en;q=0.9",
    })
    return session

async def fetch_bundle_and_extract_id() -> str:
    """Fetch bundle and extract setAnonCookies ID with caching"""
    if state.set_anon_id_cache:
        return state.set_anon_id_cache
    
    session = await create_http_session()
    try:
        response = await session.get(config.BUNDLE_URL)
        js_code = response.text
        
        pos = js_code.find('"setAnonCookies"')
        if pos == -1:
            raise RuntimeError("setAnonCookies not found")
        
        ctx = js_code[max(0, pos-150):pos]
        hex_matches = re.findall(r'"([a-f0-9]{40,50})"', ctx)
        if not hex_matches:
            raise RuntimeError("setAnonCookies ID not found")
        
        set_id = hex_matches[-1]
        state.set_anon_id_cache = set_id  # Cache the result
        return set_id
    finally:
        await session.close()

def generate_keypair() -> bytes:
    """Generate secp256k1 keypair"""
    priv = secp256k1.PrivateKey()
    return priv.private_key

async def create_anon_user(session: AsyncSession, priv_bytes: bytes) -> str:
    """Create anonymous user"""
    priv = secp256k1.PrivateKey(priv_bytes)
    pubkey_bytes = priv.pubkey.serialize(compressed=False)
    pubkey_b64 = base64.b64encode(pubkey_bytes).decode()
    
    headers = {
        "content-type": "application/json",
        "origin": config.BASE_URL,
        "referer": config.BASE_URL + "/"
    }
    
    response = await session.post(
        config.API_CREATE_ANON, 
        headers=headers, 
        json={"userPublicKey": pubkey_b64}
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to create anon user: {response.status_code} - {response.text}")
    
    try:
        data = response.json()
    except:
        raise RuntimeError(f"Invalid JSON response: {response.text}")
    
    if not data or "anonUserId" not in data:
        raise RuntimeError(f"anonUserId not found in response: {data}")
    
    return data["anonUserId"]

async def fetch_challenge(session: AsyncSession, anon_id: str) -> str:
    """Fetch challenge for signing"""
    response = await session.post(config.API_CHALLENGE, json={"anonUserId": anon_id})
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch challenge: {response.status_code} - {response.text}")
    
    try:
        data = response.json()
    except:
        raise RuntimeError(f"Invalid JSON response: {response.text}")
    
    if not data or "challenge" not in data:
        raise RuntimeError(f"Challenge not found in response: {data}")
    
    return data["challenge"]

async def generate_auth_tokens() -> Dict[str, str]:
    """Generate fresh authentication tokens with proper session setup"""
    session = await create_http_session()
    
    try:
        # Set basic cookies first
        session.cookies.update({
            "i18nextLng": "en",
            "stblid": str(uuid.uuid4())
        })
        
        # Get bundle ID
        set_anon_id = await fetch_bundle_and_extract_id()
        
        # Generate keypair and create user
        priv_bytes = generate_keypair()
        anon_id = await create_anon_user(session, priv_bytes)
        challenge_b64 = await fetch_challenge(session, anon_id)
        
        # Sign challenge
        challenge_raw = base64.b64decode(challenge_b64)
        digest = hashlib.sha256(challenge_raw).digest()
        priv = secp256k1.PrivateKey(priv_bytes)
        rec_sig = priv.ecdsa_sign_recoverable(digest, raw=True)
        sig_65, _ = priv.ecdsa_recoverable_serialize(rec_sig)
        sig_b64 = base64.b64encode(sig_65).decode()
        
        # Perform setAnonCookies action
        headers = {
            "accept": "text/x-component",
            "content-type": "application/json",
            "next-action": set_anon_id,
            "referer": config.BASE_URL + "/",
            "origin": config.BASE_URL,
        }
        payload = json.dumps([{
            "anonUserId": anon_id,
            "challenge": challenge_b64,
            "signature": sig_b64
        }], separators=(",", ":"))

        await session.post(config.BASE_URL + "/", headers=headers, data=payload)
        
        return {
            "x-anonuserid": anon_id,
            "x-challenge": challenge_b64,
            "x-signature": sig_b64
        }
    
    except Exception as e:
        raise RuntimeError(f"Auth token generation failed: {e}")
    finally:
        await session.close()

# === Statsig Pool Management ===
async def harvest_single_statsig() -> Optional[str]:
    """Harvest single statsig ID using playwright"""
    try:
        # Import here to avoid startup issues if playwright not installed
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
            )
            
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            page = await context.new_page()
            
            # Block unnecessary resources
            await page.route("**/*", lambda route: (
                route.abort() if route.request.resource_type in ["image", "font", "media"]
                else route.continue_()
            ))
            
            result = {"x-statsig-id": None}
            token_found = asyncio.Event()
            
            # Intercept requests
            async def intercept_request(route):
                if "/rest/app-chat/conversations/new" in route.request.url:
                    headers = route.request.headers
                    if "x-statsig-id" in headers:
                        result["x-statsig-id"] = headers["x-statsig-id"]
                        token_found.set()
                        await route.abort()
                        return
                await route.continue_()
            
            await page.route("**/rest/app-chat/conversations/new", intercept_request)
            
            # Navigate and interact
            await page.goto("https://grok.com/chat", wait_until="commit", timeout=30000)
            await page.wait_for_selector('textarea[aria-label="Ask Grok anything"]', timeout=30000)
            await page.type('textarea[aria-label="Ask Grok anything"]', "Hello")
            await page.keyboard.press("Enter")
            
            # Wait for token
            await asyncio.wait_for(token_found.wait(), timeout=20.0)
            
            await browser.close()
            return result.get("x-statsig-id")
            
    except Exception as e:
        print(f"Harvest error: {e}")
        return None

async def maintain_statsig_pool():
    """Background task to maintain statsig pool"""
    while True:
        try:
            pool_size = state.statsig_pool.qsize()
            
            if pool_size < config.STATSIG_REFRESH_THRESHOLD:
                print(f"Pool low ({pool_size}), harvesting...")
                
                async with state.harvesting_lock:
                    # Harvest multiple tokens concurrently
                    tasks = [harvest_single_statsig() for _ in range(5)]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, str) and result:
                            await state.statsig_pool.put(result)
                            print(f"Added statsig token: {result[:12]}...")
                
                print(f"Pool updated. Current size: {state.statsig_pool.qsize()}")
            
            # Check every 5 minutes
            await asyncio.sleep(300)
            
        except Exception as e:
            print(f"Pool maintenance error: {e}")
            await asyncio.sleep(60)

# === API Handlers ===
async def make_grok_request(messages: List[ChatMessage], statsig_id: str) -> AsyncGenerator[str, None]:
    """Make request to Grok API and stream response"""
    session = await create_http_session()
    
    try:
        # Set essential cookies
        session.cookies.update({
            "i18nextLng": "en",
            "stblid": str(uuid.uuid4())
        })
        
        # Generate fresh auth tokens
        auth_tokens = await generate_auth_tokens()
        
        # Set auth cookies
        session.cookies.update({
            "x-anonuserid": auth_tokens["x-anonuserid"],
            "x-challenge": auth_tokens["x-challenge"], 
            "x-signature": auth_tokens["x-signature"]
        })
        
        # Prepare headers
        headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.9",
            "content-type": "application/json",
            "origin": config.BASE_URL,
            "priority": "u=1, i",
            "referer": config.BASE_URL + "/",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-statsig-id": statsig_id,
            "x-xai-request-id": str(uuid.uuid4()),
        }
        
        # Convert messages to Grok format
        message_content = messages[-1].content if messages else "Hello"
        
        payload = {
            "message": message_content,
            "modelName": "grok-3"
        }
        
        print(f"Sending to Grok: {message_content}")
        
        # Make streaming request using curl_cffi's stream API
        response = await session.post(
            config.BASE_URL + "/rest/app-chat/conversations/new",
            json=payload,
            headers=headers,
            timeout=config.REQUEST_TIMEOUT,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"Grok API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        print("Starting to stream response...")
        
        # Use aiter_lines() for async line-by-line streaming
        async for line in response.aiter_lines():
            if not line:
                continue
                
            line_str = line.strip()
            if not line_str:
                continue
            
           # print(f"Raw line: {line_str[:100]}...")
            
            try:
                # Try to parse as JSON
                json_data = json.loads(line_str)
                
                # Check if this is the final chunk
                if (isinstance(json_data, dict) and 
                    json_data.get('result', {}).get('response', {}).get('isSoftStop', False)):
                    return  # End of stream
                    
                # Extract content and yield immediately
                content = extract_content_from_json(json_data)
                if content is not None:
                    yield content
                    
            except json.JSONDecodeError:
                # If not JSON, might be plain text - yield directly
                if line_str and not line_str.startswith('{'):
                    yield line_str
                
    except Exception as e:
        print(f"Request error: {e}")
        yield f"Error: {str(e)}"
    finally:
        await session.close()

def extract_content_from_json(json_data):
    """Helper function to extract content from JSON response"""
    if not isinstance(json_data, dict):
        return None
        
    # Try various content extraction paths - prioritize Grok's actual format
    paths = [
        ["result", "response", "token"],  # Grok's actual streaming format
        ["result", "message", "text"],
        ["result", "message", "content"],
        ["result", "text"],
        ["result", "content"],
        ["text"],
        ["content"],
        ["data"],
        ["choices", 0, "delta", "content"],
        ["choices", 0, "message", "content"],
        ["token"]  # Direct token field
    ]
    
    for path in paths:
        try:
            current = json_data
            for key in path:
                if isinstance(key, int):
                    current = current[key] if isinstance(current, list) and len(current) > key else None
                else:
                    current = current.get(key) if isinstance(current, dict) else None
                if current is None:
                    break
            
            if current and isinstance(current, str):
                # Don't strip for tokens - preserve spaces!
                return current
        except (KeyError, IndexError, TypeError):
            continue
    
    return None

async def convert_to_openai_format(grok_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Convert Grok response to OpenAI format - ZERO buffering"""
    try:
        async for chunk in grok_stream:
            # Send EVERYTHING immediately - no filtering
            if chunk is not None:
                openai_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "grok-3",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                
                # Stream immediately - no delays
                yield f"data: {json.dumps(openai_chunk)}\n\n"
        
        # Send completion signal
        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk", 
            "created": int(time.time()),
            "model": "grok-3",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
# === FastAPI Setup ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Grok proxy server...")
    
    # Initial statsig pool harvest
    print("Harvesting initial statsig pool...")
    tasks = [harvest_single_statsig() for _ in range(config.STATSIG_POOL_SIZE)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, str) and result:
            await state.statsig_pool.put(result)
    
    print(f"Initial pool size: {state.statsig_pool.qsize()}")
    
    # Start background task
    asyncio.create_task(maintain_statsig_pool())
    
    yield
    
    # Shutdown
    print("Shutting down...")

app = FastAPI(title="Grok Proxy", lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    async with state.request_semaphore:
        try:
            # Get statsig ID from pool
            if state.statsig_pool.empty():
                raise HTTPException(status_code=503, detail="No available tokens")
            
            statsig_id = await state.statsig_pool.get()
            
            try:
                # Make request to Grok
                grok_stream = make_grok_request(request.messages, statsig_id)
                openai_stream = convert_to_openai_format(grok_stream)
                
                return StreamingResponse(
                    openai_stream,
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache"}
                )
                
            finally:
                # Return token to pool
                await state.statsig_pool.put(statsig_id)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "statsig_pool_size": state.statsig_pool.qsize(),
        "timestamp": int(time.time())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)