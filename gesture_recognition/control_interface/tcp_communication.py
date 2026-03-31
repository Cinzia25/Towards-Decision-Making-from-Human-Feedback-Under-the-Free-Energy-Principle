import socket

class TCPCommunication:
    def __init__(self, host='10.160.164.192', port=5005):   # Robot IP
        # Store connection parameters
        self.host = host
        self.port = port

        # Create TCP socket (IPv4, stream-oriented)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Attempt to establish connection with server (robot)
            self.socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
        except Exception as e:
            # If connection fails, invalidate socket
            print(f"Failed to connect to {self.host}:{self.port} - {e}")
            self.socket = None

    def sending_data(self, command: str) -> None:
        # Check if socket is valid
        if self.socket:
            try:
                # Encode string command to ASCII and send it
                self.socket.sendall(command.encode('ascii'))
                print(f"Sent command: {command}")
            except Exception as e:
                # Transmission failure (e.g., broken pipe)
                print(f"Failed to send command: {e}")
        else:
            # No active connection
            print("No TCP connection established.")

    def close(self):
        # Close socket if it exists
        if self.socket:
            self.socket.close()
            print("TCP connection closed.")