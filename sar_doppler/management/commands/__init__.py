import logging
import logging.handlers


def worker_init(queue):
    """Set up each worker process to send log records to the shared queue.

    Replaces any inherited handlers (which may carry a locked threading.RLock
    from the parent process after fork) with a QueueHandler so that all log
    records are forwarded to the main process for safe, serialised file I/O.
    """
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(logging.INFO)
