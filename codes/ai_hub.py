from flask import Flask, request
import json
import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)
_global_dict = {}


def _init():
    global _global_dict
    _global_dict = {}


def set_value(name, value):
    _global_dict[name] = value


def get_value(name, defValue=None):
    try:
        return _global_dict[name]
    except KeyError:
        return defValue


app = Flask("tccapi")


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


@app.route("/shutdown", methods=["GET", "POST"])
def shutdown():
    shutdown_server()
    return "Server shutting down..."


@app.route("/tccapi", methods=["GET", "POST"])
def tccapi():
    ret = ""
    if request.method == "POST":
        data = request.get_data()
        if data == b"exit" or data == "exit":
            print("Server shutting down...")
            shutdown_server()
            return "Server shutting down..."
        inferserver = get_value("inferserver")
        data_pred = inferserver.pre_process(request)
        ret = inferserver.predict(data_pred)
        ret = inferserver.post_process(ret)
        if not isinstance(ret, str):
            ret = str(ret)
        # print("return: ", ret)
    else:
        print(
            "please use post request. such as : curl localhost:8080/tccapi -X POST -d '{\"img\"/:2}'"
        )
    return ret


@app.errorhandler(500)
def internal_error(error):
    print("Terminated: Error Caught!")
    shutdown_server()
    return "[]"


class inferServer(object):
    def __init__(self, model):
        self.model = model
        print("init_Server")
        _init()
        set_value("inferserver", self)

    def pre_process(self, request):
        data = request.get_data()
        return data

    def post_process(self, data):
        return data

    def predict(self, data):
        data = self.model(data)
        return data

    def run(self, ip="127.0.0.1", port=8080, debuge=False):
        app.run(ip, port, debuge)
