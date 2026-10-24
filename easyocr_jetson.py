# Runs on the Jetson. Receives webcam frames from the laptop, runs EasyOCR,
# and sends the results back.
import socket, struct, json
import cv2, numpy as np, easyocr

PORT = 5000
reader = easyocr.Reader(
    ['en'],
    gpu=True,
    model_storage_directory='/home/uwdrones/.EasyOCR/model',
    user_network_directory='/home/uwdrones/.EasyOCR/user_network',
    download_enabled=False,   # models were copied over manually
)

def recv_exact(conn, n):
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', PORT))
server.listen(1)
print(f"Waiting for laptop on port {PORT}...")

while True:
    conn, addr = server.accept()
    print("Connected:", addr)
    while True:
        header = recv_exact(conn, 4)
        if header is None:
            break
        size = struct.unpack('>I', header)[0]
        jpeg = recv_exact(conn, size)
        if jpeg is None:
            break
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

        results = []
        for box, text, conf in reader.readtext(frame):
            results.append({
                "box": [[int(x), int(y)] for x, y in box],
                "text": text,
                "conf": float(conf),
            })

        reply = json.dumps(results).encode()
        conn.sendall(struct.pack('>I', len(reply)) + reply)
    conn.close()
    print("Laptop disconnected")
