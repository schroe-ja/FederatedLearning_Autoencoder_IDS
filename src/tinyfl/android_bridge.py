import grpc

from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Event, Lock
from http.server import BaseHTTPRequestHandler, HTTPServer

from flwr.proto import transport_pb2 as pb
from flwr.proto import transport_pb2_grpc as rpc


requests = Queue()
responses = Queue()
connected = Event()
lock = Lock()


class AndroidBridge(rpc.FlowerServiceServicer):

    def Join(self, request_iterator, context):
        if not lock.acquire(blocking=False):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Android already connected",
            )

        connected.set()

        try:
            while context.is_active():
                try:
                    instruction = requests.get(timeout=1)
                except Empty:
                    continue

                yield instruction
                responses.put(next(request_iterator))

        except StopIteration:
            pass

        finally:
            connected.clear()
            lock.release()


class Handler(BaseHTTPRequestHandler):

    def do_POST(self):
        if not connected.wait(timeout=60):
            self.send_error(503, "Android not connected")
            return

        data = self.rfile.read(
            int(self.headers["Content-Length"])
        )

        requests.put(pb.ServerMessage.FromString(data))

        try:
            result = responses.get(timeout=600)
        except Empty:
            self.send_error(504, "Android timeout")
            return

        payload = result.SerializeToString()

        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    server = grpc.server(ThreadPoolExecutor(max_workers=4))

    rpc.add_FlowerServiceServicer_to_server(
        AndroidBridge(), server
    )

    server.add_insecure_port("0.0.0.0:8081")
    server.start()

    HTTPServer(("127.0.0.1", 8082), Handler).serve_forever()