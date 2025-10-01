import asyncio
import websockets
import json
import logging
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderbookWebSocketClient:
    """WebSocket client for streaming L2 orderbook data"""
    
    def __init__(self, url: str, on_message_callback: Callable):
        """
        Initialize WebSocket client
        
        Args:
            url: WebSocket URL to connect to
            on_message_callback: Callback function to handle incoming messages
        """
        self.url = url
        self.on_message_callback = on_message_callback
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_running = False
    
    async def connect(self):
        """Connect to WebSocket and start receiving messages"""
        self.is_running = True
        retry_count = 0
        max_retries = 5
        
        while self.is_running and retry_count < max_retries:
            try:
                logger.info(f"Connecting to WebSocket: {self.url}")
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    logger.info("WebSocket connected successfully")
                    retry_count = 0  # Reset retry count on successful connection
                    
                    # Listen for messages
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        try:
                            data = json.loads(message)
                            # Call the callback with parsed data
                            self.on_message_callback(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse message: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            
            except websockets.exceptions.WebSocketException as e:
                retry_count += 1
                logger.error(f"WebSocket error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    logger.error("Max retries reached. Giving up.")
                    self.is_running = False
                    
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.is_running = False
                
    def close(self):
        """Close the WebSocket connection"""
        self.is_running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        logger.info("WebSocket closed")
