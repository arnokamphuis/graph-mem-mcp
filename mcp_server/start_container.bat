@echo off

REM Build the container
echo Building the MCP Graph Memory container...
podman build -t graph-mem-mcp .

REM Run the container with HTTP server and MCP support
echo Starting the container on port 10642...
podman run -d ^
  --name graph-mem-mcp ^
  -p 10642:10642 ^
  -v graph-mem-data:/data ^
  graph-mem-mcp ^
  python main.py --mcp-with-http

echo Container started!
echo Access the visualization at: http://localhost:10642
echo API endpoints available at: http://localhost:10642/docs
echo.
echo To stop the container: podman stop graph-mem-mcp
echo To remove the container: podman rm graph-mem-mcp
echo To view logs: podman logs graph-mem-mcp
pause
