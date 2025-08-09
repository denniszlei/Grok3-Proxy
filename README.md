# Grok3 Proxy API

A high-performance streaming proxy server that provides OpenAI-compatible API endpoints for educational and research purposes.

## ⚠️ Important Disclaimer

> **FOR EDUCATIONAL PURPOSES ONLY**
> 
> This project is created solely for educational and research purposes to demonstrate API proxy patterns, streaming implementations, and async programming techniques in Python. 
>
> - **NOT affiliated** with Grok, xAI, or any related entities
> - **NOT intended** for production use or commercial purposes  
> - **Reverse engineering** of proprietary services may violate Terms of Service
> - **Users are responsible** for compliance with applicable laws and ToS
> - **Use at your own risk** - the authors disclaim all liability

## 🚀 Features

- **Real-time streaming** - Zero-buffering token streaming
- **OpenAI Compatible** - Drop-in replacement for OpenAI API endpoints
- **Browser Impersonation** - Advanced anti-detection using curl_cffi
- **Async Architecture** - High-performance FastAPI with asyncio
- **Connection Pooling** - Efficient resource management
- **Auto Token Management** - Automated authentication handling

## 📋 Requirements

- Python 3.8+
- Playwright (for token harvesting)
- curl_cffi
- FastAPI
- aiohttp

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/idekubaguscom/Grok3-Proxy.git
cd grok3-proxy

# Install dependencies
pip install -r requirements.txt

# Install playwright browsers
playwright install chromium
```

## 🔧 Configuration

The proxy uses a configuration class that can be customized:

```python
@dataclass
class Config:
    STATSIG_POOL_SIZE = 5
    STATSIG_REFRESH_THRESHOLD = 2
    MAX_CONCURRENT_REQUESTS = 100
    REQUEST_TIMEOUT = 30
    BASE_URL = "https://grok.com"
```

## 🚀 Usage

### Start the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will start on `http://localhost:8000`

### API Endpoints

#### Chat Completions (OpenAI Compatible)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grok-3",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": true
  }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

### Response Format

The API returns OpenAI-compatible streaming responses:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"grok-3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"grok-3","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: [DONE]
```

## 🏗️ Architecture

### Core Components

1. **Authentication System** - Handles token generation and management
2. **Statsig Pool** - Maintains pool of authentication tokens using Playwright
3. **Streaming Engine** - Real-time response streaming with zero buffering
4. **Connection Management** - Efficient session pooling and reuse

### Request Flow

```
Client Request → FastAPI → Auth Token Pool → Target Service → Stream Parser → Client Response
```

### Key Technical Features

- **Browser Impersonation**: Uses curl_cffi with Chrome 120 fingerprints
- **Concurrent Processing**: Semaphore-based request limiting
- **Memory Efficient**: Streaming without buffering large responses
- **Auto Recovery**: Automatic token refresh and error handling

## 🔧 Development

### Project Structure

```
grok3-proxy/
├── main.py              # Main application file
├── requirements.txt     # Dependencies
├── README.md           # This file
└── .gitignore         # Git ignore rules
```

### Key Functions

- `make_grok_request()` - Core streaming request handler
- `maintain_statsig_pool()` - Background token management  
- `convert_to_openai_format()` - Response format conversion
- `harvest_single_statsig()` - Individual token acquisition

### Streaming Implementation

```python
# Zero-buffering streaming
async for line in response.aiter_lines():
    content = extract_content_from_json(json.loads(line))
    if content is not None:
        yield content  # Immediate yield - no filtering
```

## 🐛 Troubleshooting

### Common Issues

**Connection Errors**
```
Request error: 'Response' object has no attribute 'iter_chunked'
```
- Ensure curl_cffi is properly installed
- Check that you're using `aiter_lines()` not `iter_chunked()`

**Token Pool Empty**
```  
HTTPException: 503 No available tokens
```
- Wait for background token harvesting to complete
- Check Playwright browser installation
- Verify network connectivity

**Streaming Not Working**
- Disable any reverse proxy buffering (nginx, cloudflare)
- Check client supports chunked transfer encoding
- Verify response headers include anti-buffering settings

## 📊 Performance

### Benchmarks

- **Latency**: ~50ms first token
- **Throughput**: 100+ concurrent requests  
- **Memory**: <100MB base usage
- **CPU**: Single core sufficient for moderate load

### Optimization Tips

1. **Increase pool size** for higher concurrency
2. **Tune semaphore limits** based on target service capacity  
3. **Monitor token refresh rate** to prevent exhaustion
4. **Use HTTP/2** connection pooling when possible

## 🔒 Security Considerations

- **No API keys stored** - All authentication is ephemeral
- **Request isolation** - Each request uses separate token  
- **Rate limiting** - Built-in concurrent request limits
- **Input validation** - Sanitized request parameters

## 🤝 Contributing

This is an educational project. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Submit a pull request

Please ensure all contributions align with the educational purpose of this project.

## 📝 License

This project is provided as-is for educational purposes. No warranty or support is provided.

## 🙏 Acknowledgments

- FastAPI for the excellent async web framework
- curl_cffi for advanced HTTP client capabilities  
- Playwright for browser automation
- The Python asyncio ecosystem

---

**Remember**: This project is for learning and research only. Always respect Terms of Service and applicable laws when working with web APIs and services.
