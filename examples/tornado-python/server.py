import os
import os.path
import logging
import tornado.websocket
import tornado.web
import tornado.ioloop
import threading

from perspective import Table, PerspectiveManager, PerspectiveTornadoHandler


here = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(
    here, "..", "..", "node_modules", "superstore-arrow", "superstore.arrow"
)


def perspective_thread(manager):
    """Perspective application thread starts its own tornado IOLoop, and
    adds the table with the name "data_source_one", which will be used
    in the front-end."""
    psp_loop = tornado.ioloop.IOLoop()
    manager.set_loop_callback(psp_loop.add_callback)
    with open(file_path, mode="rb") as file:
        table = Table(file.read(), index="Row ID")
        manager.host_table("data_source_one", table)
        manager.host_view("view_one", table.view())
    psp_loop.start()


def make_app():
    manager = PerspectiveManager()

    thread = threading.Thread(target=perspective_thread, args=(manager,))
    thread.daemon = True
    thread.start()

    return tornado.web.Application(
        [
            (
                r"/websocket",
                PerspectiveTornadoHandler,
                {"manager": manager, "check_origin": True},
            ),
            (
                r"/node_modules/(.*)",
                tornado.web.StaticFileHandler,
                {"path": "../../node_modules/@finos/"},
            ),
            (
                r"/(.*)",
                tornado.web.StaticFileHandler,
                {"path": "./", "default_filename": "index.html"},
            ),
        ]
    )


if __name__ == "__main__":
    app = make_app()
    app.listen(8080)
    logging.critical("Listening on http://localhost:8080")
    loop = tornado.ioloop.IOLoop.current()
    loop.start()
