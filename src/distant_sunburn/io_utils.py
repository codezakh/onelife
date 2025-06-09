import json
from typing import IO, Any
from typing_extensions import Self
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import (
    Collection,
    Generator,
    Generic,
    Iterator,
    MutableMapping,
    Optional,
    Type,
)

from pydantic import BaseModel
from tqdm.auto import tqdm
from ulid import ULID
import collections.abc

from .typing_utils import BaseModelT


class JsonlIoHandler(collections.abc.Iterable[dict]):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def __iter__(self) -> Iterator[dict]:
        with open(self.file_path, "r") as f:
            for line in f:
                yield json.loads(line)

    def append(self, data: dict) -> None:
        """Appends a dictionary as a new line in the JSONL file."""
        with open(self.file_path, "a") as f:
            json_str = json.dumps(data)
            f.write(json_str + "\n")

    def read_all(self, progress: Optional[bool] = False) -> list[dict]:
        """Reads all dictionaries from the JSONL file. Optional progress indicator."""
        data: list[dict] = []
        iterator = self.__iter__()
        if progress:
            iterator = tqdm(iterator, desc="Reading JSONL")  # type: ignore
        for item in iterator:
            data.append(item)
        return data

    def read_n(self, n: int) -> list[dict]:
        """Reads the first n dictionaries from the JSONL file."""
        data: list[dict] = []
        iterator = self.__iter__()
        for _ in range(n):
            data.append(next(iterator))
        return data


class PydanticJSONLinesWriter(Generic[BaseModelT]):
    """
    Write Pydantic model instances to a JSONL file.

    This class can be used as a context manager to ensure that the file is closed
    after the writer is done. This is useful when doing a lot of writes in a tight loop
    and you want to avoid opening and closing the file multiple times.

    Usage:
    ```python
    writer = PydanticJSONLinesWriter("file.jsonl")
    writer(model_instance)
    writer.write_batch([model_instance1, model_instance2])
    with writer:
        for n in range(1000):
            model_instance = get_model_instance()
            writer(model_instance)
    ```
    """

    def __init__(self, file_path: str | Path, mode: str = "a"):
        self.file_path = file_path
        self._file: Optional[IO[Any]] = None
        self._mode = mode

    def __enter__(self) -> Self:
        self._file = open(self.file_path, mode=self._mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __call__(self, serializable: BaseModelT) -> None:
        if self._file is None:
            with open(self.file_path, "a") as f:
                f.write(serializable.model_dump_json() + "\n")
        else:
            self._file.write(serializable.model_dump_json() + "\n")

    def write_batch(
        self, serializables: Collection[BaseModelT], mode: str = "a"
    ) -> None:
        if self._file is None:
            with open(self.file_path, mode) as f:
                for serializable in serializables:
                    f.write(serializable.model_dump_json() + "\n")
        else:
            for serializable in serializables:
                self._file.write(serializable.model_dump_json() + "\n")


class PydanticJSONLinesReader(collections.abc.Iterable[BaseModelT]):
    def __init__(self, file_path: str | Path, model: Type[BaseModelT]):
        self.file_path = file_path
        self.model = model

    def __iter__(self) -> Iterator[BaseModelT]:
        with open(self.file_path, "r") as f:
            for line in f:
                yield self.model.model_validate_json(line)

    def read_all(self) -> list[BaseModelT]:
        return list(self)

    def read_n(self, n: int) -> list[BaseModelT]:
        data: list[BaseModelT] = []
        iterator = self.__iter__()
        for _ in range(n):
            data.append(next(iterator))
        return data


def extract_code_in_markdown_backticks(model_output: str) -> str:
    outputlines = model_output.split("\n")
    # Find the first line that starts with ```
    opening_backticks_idx = next(
        (i for i, line in enumerate(outputlines) if line.startswith("```")), None
    )
    if opening_backticks_idx is None:
        # We don't know what to do, so just return the whole thing.
        return "\n".join(outputlines)

    # Find the line that contains the closing code block.
    closing_backticks_idx = next(
        (
            i
            for i, line in enumerate(outputlines[opening_backticks_idx:])
            if line.endswith("```")
        ),
        None,
    )

    if closing_backticks_idx is None:
        # We don't know what to do, so just return the whole thing.
        return "\n".join(outputlines)

    # If there isn't any code between them, return the whole thing.
    if opening_backticks_idx + 1 >= closing_backticks_idx:
        return "\n".join(outputlines)

    return "\n".join(outputlines[opening_backticks_idx + 1 : closing_backticks_idx])


class BufferedPydanticJSONLinesWriter(Generic[BaseModelT]):
    def __init__(
        self, file_path: str | Path, buffer_size: int = 50, flush_interval: float = 1.0
    ):
        self.file_path = Path(file_path)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.queue: Queue[BaseModelT] = Queue()
        self.writer = PydanticJSONLinesWriter(self.file_path)
        self.stop_event = Event()
        self.worker = None

    def _start_worker(self) -> None:
        if not self.worker or not self.worker.is_alive():
            self.worker = Thread(target=self._worker_loop, daemon=True)
            self.worker.start()

    def _worker_loop(self) -> None:
        buffer: list[BaseModelT] = []
        last_flush_time = time.time()

        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=self.flush_interval)
                buffer.append(item)
            except Empty:
                pass

            time_since_last_flush = time.time() - last_flush_time
            if (
                len(buffer) >= self.buffer_size
                or time_since_last_flush >= self.flush_interval
            ):
                self.writer.write_batch(buffer)
                buffer.clear()
                last_flush_time = time.time()

    def __call__(self, item: BaseModelT) -> None:
        self.queue.put(item)

    def write_batch(self, items: Collection[BaseModelT]) -> None:
        for item in items:
            self.queue.put(item)

    def close(self) -> None:
        # Flush any remaining items in the buffer by writing them to disk
        while not self.queue.empty():
            buffer = []
            while not self.queue.empty():
                try:
                    buffer.append(self.queue.get_nowait())
                except Empty:
                    break
            if buffer:
                self.writer.write_batch(buffer)
        self.stop_event.set()
        if self.worker:
            self.worker.join()

    def __enter__(self) -> "BufferedPydanticJSONLinesWriter[BaseModelT]":
        self._start_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
