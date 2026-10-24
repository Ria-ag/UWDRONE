# Runs on the laptop. Sends webcam frames to the Jetson and shows the OCR results.
# Runs forever. Stop with Ctrl+C in the terminal.
# Usage: python laptop_client.py [jetson-ip]
import socket, struct, json, sys, time
import cv2

JETSON_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.50.2"  # Jetson IP on the ethernet cable
PORT = 5000

def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Jetson closed the connection")
        data += chunk
    return data

cam = cv2.VideoCapture(0)

while True:
    try:
        sock = socket.create_connection((JETSON_IP, PORT))
        print("Connected to Jetson")
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.1)
                continue
            frame = cv2.resize(frame, (640, 480))

            # send frame as JPEG
            jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            sock.sendall(struct.pack('>I', len(jpeg)) + jpeg)

            # receive results
            size = struct.unpack('>I', recv_exact(sock, 4))[0]
            results = json.loads(recv_exact(sock, size))

            # draw boxes and text, and print
            for r in results:
                pts = [tuple(p) for p in r["box"]]
                for i in range(4):
                    cv2.line(frame, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
                x, y = pts[0]
                cv2.putText(frame, r["text"], (x, max(y - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f'{r["text"]} ({r["conf"]:.2f})')

            cv2.imshow("Jetson OCR", frame)
            cv2.waitKey(1)  # needed to refresh the window; does not wait or quit
    except (ConnectionError, OSError) as e:
        print("Connection problem:", e, "- retrying in 3s")
        time.sleep(3)
