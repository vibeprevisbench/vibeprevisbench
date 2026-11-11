"""ModelFixture of the frozen harness tests (tests/test_harness.py), verbatim: a local HTTP model whose answers
are given in advance. The replay uses it with one answer, the delivered solution.py."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import ChatModel


class ModelFixture:
    def __init__(self, answers, first_error=False):
        self.answers = list(answers)
        self.requests = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fixture.requests.append(request)
                if first_error and len(fixture.requests) == 1:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b'{"error":"test infrastructure failure"}')
                    return
                message = fixture.answers.pop(0)
                response = {"choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                                      "prompt_tokens_details": {"cached_tokens": 40},
                                      "completion_tokens_details": {"reasoning_tokens": 5}}}
                if 'input' in request:
                    output=[]
                    if message.get('content'):
                        output.append({'type':'message','role':'assistant','content':[{'type':'output_text','text':message['content']}]})
                    for tool in message.get('tool_calls') or []:
                        output.append({'type':'function_call','call_id':tool['id'],'name':tool['function']['name'],
                                       'arguments':tool['function']['arguments']})
                    response={'status':'completed','output':output,
                              'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120,
                                       'input_tokens_details':{'cached_tokens':40},'output_tokens_details':{'reasoning_tokens':5}}}
                if '_raw_response' in message:
                    response = message['_raw_response']
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def model(self, protocol="chat_completions"):
        return ChatModel(f'http://127.0.0.1:{self.server.server_port}/v1', 'TEST_ONLY', protocol=protocol)
