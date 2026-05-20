import logging
import logging.handlers


def worker_init(log_queue):
    """Initialise logging in a pool worker.

    Clears any handlers inherited from the parent process (which may carry
    locks in a permanently-held state after fork) and installs a
    QueueHandler that forwards all log records to the main-process listener
    via a multiprocessing.Queue.  No file I/O happens inside the worker, so
    Lustre/NFS flock issues cannot block logging.
    """
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)
