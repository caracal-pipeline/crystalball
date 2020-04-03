from collections import defaultdict
from pprint import pprint
from timeit import default_timer
from threading import Event, Thread, Lock
import time
import sys

import dask
import dask.array as da
from dask.base import unpack_collections
from dask.callbacks import Callback
from dask.diagnostics.progress import format_time
from dask.optimization import key_split
from dask.utils import ignoring


class TaskData(object):
    __slots__ = ("total", "completed", "time_sum")

    def __init__(self, completed=0, total=0, time_sum=0.0):
        self.completed = completed
        self.total = total
        self.time_sum = time_sum


    def __iadd__(self, other):
        self.completed += other.completed
        self.total += other.total
        self.time_sum += other.time_sum
        return self

    def __add__(self, other):
        return TaskData(self.completed + other.completed,
                        self.total + other.total,
                        self.time_sum + other.time_sum)

    def __repr__(self):
        return "TaskData(%s, %s, %s)" % (self.completed,
                                         self.total,
                                         self.time_sum)

    __str__ = __repr__


def update_bar(elapsed, prev_completed, prev_estimated, pb):
    total = 0
    completed = 0
    estimated = 0.0
    time_guess = 0.0

    # update
    with pb.lock:
        for k, v in pb.task_data.items():
            total += v.total
            completed += v.completed
            remaining = v.total - v.completed

            if v.completed > 0:
                avg_time = v.time_sum / v.completed
                estimated += avg_time * v.total
                time_guess += v.time_sum

    if completed != prev_completed:
        estimated = estimated * elapsed / time_guess 
    else:
        estimated = prev_estimated

    fraction = completed / total
    bar = "#" * int(pb._width * fraction)
    percent = int(100 * fraction)
    msg = "\r[{0:<{1}}] | {2}% Completed | {3} | ~{4}".format(
        bar, pb._width, percent,
        format_time(elapsed),
        "???" if estimated == 0.0 else format_time(estimated))
    with ignoring(ValueError):
        pb._file.write(msg)
        pb._file.flush()

    return completed, estimated


def timer_func(pb):
    start = default_timer()

    while pb.running.is_set():
        elapsed = default_timer() - start
        prev_completed = 0
        prev_estimated = 0.0

        if elapsed > pb._minimum:
            prev_completed, prev_estimated = update_bar(elapsed, prev_completed, prev_estimated, pb)

        time.sleep(pb._dt)
    

class ProgressBar(Callback):
    def __init__(self, collections=(), minimum=0, width=40, dt=0.1, out=None):
        if out is None:
            out = sys.stdout
        self._minimum = minimum
        self._width = width
        self._dt = dt
        self._file = out
        self.lock = Lock()
        self.last_duration = 0
        self.collections, _ = unpack_collections(collections)

        self.task_start = {}
        self.task_data = defaultdict(TaskData)


    def _start(self, dsk):
        for k, v in dsk.items():
            self.task_data[key_split(k)].total += 1

        self.running = running = Event()
        self.running.set()
        self.thread = Thread(target=timer_func, args=(self,))
        self.daemon = True
        self.thread.start()

    def _finish(self, dsk, state, errored):
        self.running.clear()

    def _pretask(self, key, dsk, state):
        with self.lock:
            self.task_start[key] = default_timer()

    def _posttask(self, key, result, dsk, state, worker_id):
        with self.lock:
            td = self.task_data[key_split(key)]
            td.time_sum += default_timer() - self.task_start.pop(key)
            td.completed += 1
