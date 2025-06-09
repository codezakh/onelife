import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)

from distant_sunburn.io_utils import (
    BaseModelT,
    BufferedPydanticJSONLinesWriter,
    PydanticJSONLinesReader,
)
from distant_sunburn.typing_utils import Error, Option, Result
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

T = TypeVar("T")


class PydanticJsonLinesFileTarget(Generic[BaseModelT]):
    def __init__(
        self,
        file_path: Path | str,
        model: Type[BaseModelT],
        completion_marker_path: Optional[Path] = None,
    ):
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        self.file_path = file_path
        self.model = model
        self.read_only = False
        self.completion_marker_path = completion_marker_path or file_path.with_suffix(
            ".completed"
        )

    def load(self) -> list[BaseModelT]:
        if not self.file_path.exists():
            logger.warning(f"File {self.file_path} does not exist")
            return []
        reader = PydanticJSONLinesReader(self.file_path, self.model)
        items = list(reader)
        logger.info(f"Loaded {len(items)} {self.model.__name__} from {self.file_path}")
        return items

    def save(self, data: Iterable[BaseModelT]):
        if self.read_only:
            raise ValueError("Target is read-only")
        self.file_path.unlink(missing_ok=True)
        with open(self.file_path, "a") as f:
            for idx, item in enumerate(data):
                f.write(item.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"Saved {idx + 1} {self.model.__name__} to {self.file_path}")

    def extend(self, data: Iterable[BaseModelT]):
        if self.read_only:
            raise ValueError("Target is read-only")
        with open(self.file_path, "a") as f:
            for item in data:
                f.write(item.model_dump_json() + "\n")

    @property
    def backup_path(self) -> Path:
        return self.file_path.with_suffix(".bak")

    def replace(self, data: Iterable[BaseModelT]):
        if self.read_only:
            raise ValueError("Target is read-only")

        shutil.copy(self.file_path, self.backup_path)
        logger.info(f"Created backup at {self.backup_path}")
        self.file_path.unlink(missing_ok=True)
        self.save(data)

    def delete(self):
        if self.read_only:
            raise ValueError("Target is read-only")
        self.file_path.unlink(missing_ok=True)
        self.completion_marker_path.unlink(missing_ok=True)
        logger.warning(f"Deleted {self.file_path} and {self.completion_marker_path}")

    def mark_completed(self):
        logger.info(f"Marking {self.file_path} as completed")
        if self.read_only:
            raise ValueError("Target is read-only")
        self.completion_marker_path.touch()

    def is_completed(self) -> bool:
        return self.completion_marker_path.exists()

    @contextmanager
    def buffered_writer(
        self,
    ) -> Generator[BufferedPydanticJSONLinesWriter[BaseModelT], None, None]:
        if self.read_only:
            raise ValueError("Target is read-only")
        with BufferedPydanticJSONLinesWriter(self.file_path) as writer:
            yield writer


class JsonLinesFileTarget:
    def __init__(
        self, file_path: Path | str, completion_marker_path: Optional[Path] = None
    ):
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        self.file_path = file_path
        self.completion_marker_path = completion_marker_path or file_path.with_suffix(
            ".completed"
        )

    def load(self) -> Sequence[dict]:
        with open(self.file_path, "r") as f:
            return [json.loads(line) for line in f]

    def save(self, data: Sequence[dict]):
        with open(self.file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def extend(self, data: Sequence[dict]):
        with open(self.file_path, "a") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def delete(self):
        self.file_path.unlink(missing_ok=True)
        self.completion_marker_path.unlink(missing_ok=True)

    def is_completed(self) -> bool:
        return self.completion_marker_path.exists()

    def mark_completed(self) -> None:
        self.completion_marker_path.touch()


class JsonFileTarget:
    def __init__(self, file_path: Path | str):
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        self.file_path = file_path

    def load(self) -> dict:
        with open(self.file_path, "r") as f:
            return json.load(f)

    def save(self, data: dict | list[dict] | list[str] | list[float] | list[int]):
        with open(self.file_path, "w") as f:
            json.dump(data, f)

    def delete(self):
        self.file_path.unlink(missing_ok=True)

    def is_completed(self) -> bool:
        return self.file_path.exists()

    def mark_completed(self) -> None:
        self.file_path.touch()


class Node(BaseModel):
    id: Union[ULID, str] = Field(union_mode="left_to_right")
    name: str
    parents: list["Node"]

    model_config = ConfigDict(frozen=True)

    def linearize(self) -> Generator["Node", None, None]:
        yield self
        for parent in self.parents:
            yield from parent.linearize()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


AncestorT = TypeVar("AncestorT")
AncestorT_co = TypeVar("AncestorT_co", covariant=True)


class AncestorLookupSuccess(BaseModel, Generic[AncestorT]):
    tag: Literal["success"] = "success"
    ancestor: AncestorT
    matches: dict[Node, bool] = Field(default_factory=dict)


class AncestorLookupFailure(BaseModel):
    tag: Literal["failure"] = "failure"
    reason: str
    matches: dict[Node, bool] = Field(default_factory=dict)


class NodeMapper(Protocol[AncestorT_co]):
    def __call__(self, node: Node) -> AncestorT_co: ...


class AdmitsNode(Protocol):
    """
    To make a class representable as a node, it must fulfill this protocol.

    Example:
    class SomeData(BaseModel):
        value: str

    class SomeDataNode(SomeData):
        parents: list["Node"]
        node_name: Optional[str] = None

        def as_node(self) -> Node:
            return Node(id=self.value, name="spoingus", parents=self.parents)

    """

    parents: list["Node"]
    node_name: Optional[str] = None

    def as_node(self) -> Node:
        """
        Represent the class as a node. Used by downstream code
        to fill the .parent attribute of a child node.
        """
        ...


AdmitsNodeT = TypeVar("AdmitsNodeT", bound=AdmitsNode)

SourceNodeValueT = TypeVar("SourceNodeValueT")


class AdmitNodesCollection(Sequence[AdmitsNodeT]):
    def __init__(self, stream: Iterable[AdmitsNodeT]):
        self.sequence = list(stream)
        self.parent_to_children: dict[Node, list[AdmitsNodeT]] = dict()
        for item in self.sequence:
            for parent in item.parents:
                try:
                    self.parent_to_children[parent].append(item)
                except KeyError:
                    self.parent_to_children[parent] = [item]

    def __iter__(self) -> Iterator[AdmitsNodeT]:
        return iter(self.sequence)

    def __getitem__(self, index: int) -> AdmitsNodeT:
        return self.sequence[index]

    def __len__(self) -> int:
        return len(self.sequence)

    def contains_child_of(self, source_node: Node) -> bool:
        return source_node in self.parent_to_children

    def get_children_of(self, source_node: Node) -> list[AdmitsNodeT]:
        return self.parent_to_children[source_node]

    def partition_source_nodes_by_presence(
        self, source_nodes: Mapping[Node, SourceNodeValueT]
    ) -> tuple[dict[Node, SourceNodeValueT], dict[Node, SourceNodeValueT]]:
        completed_source_nodes = {
            k: v for k, v in source_nodes.items() if self.contains_child_of(k)
        }
        pending_source_nodes = {
            k: v for k, v in source_nodes.items() if not self.contains_child_of(k)
        }
        return completed_source_nodes, pending_source_nodes

    def append(self, item: AdmitsNodeT):
        self.sequence.append(item)
        for parent in item.parents:
            try:
                self.parent_to_children[parent].append(item)
            except KeyError:
                self.parent_to_children[parent] = [item]


def get_ancestor(
    node: AdmitsNode | Node, mapper: NodeMapper[AncestorT]
) -> Option[AncestorLookupSuccess[AncestorT], AncestorLookupFailure]:
    parent_nodes = node.parents
    if not parent_nodes:
        return Error(
            AncestorLookupFailure(reason="Node has no parents."),
        )

    matches: dict[Node, bool] = dict()
    for parent in parent_nodes:
        for ancestor_node in parent.linearize():
            try:
                ancestor = mapper(ancestor_node)
            except KeyError:
                matches[ancestor_node] = False
                continue
            else:
                matches[ancestor_node] = True
                return Result(
                    AncestorLookupSuccess(ancestor=ancestor, matches=matches),
                )

    return Error(
        AncestorLookupFailure(
            reason="No ancestor found in mapper.",
            matches=matches,
        ),
    )


class NoAncestorFound(Exception):
    def __init__(self, reason: str):
        self.reason = reason

    def __str__(self) -> str:
        return self.reason

    def __repr__(self) -> str:
        return f"NoAncestorFound(reason={self.reason})"


def get_ancestor_throw_on_failure(
    node: AdmitsNode | Node, mapper: NodeMapper[AncestorT]
) -> AncestorT:
    result = get_ancestor(node, mapper)
    if result.is_err():
        raise NoAncestorFound(result.unwrap_err().reason)
    return result.unwrap().ancestor


class HasUlid(Protocol):
    ulid: ULID


HasUlidT = TypeVar("HasUlidT", bound=HasUlid)


class UlidMapper(Generic[HasUlidT]):
    def __init__(self, stream: Iterable[HasUlidT], node_name: Optional[str] = None):
        self.stream = stream
        self.lookup_table = {_.ulid: _ for _ in stream}
        self.node_name = node_name

    def __call__(self, node: Node) -> HasUlidT:
        found = self.lookup_table[node.id]  # type: ignore
        if self.node_name is None:
            return found
        else:
            if self.node_name != node.name:
                raise KeyError(
                    f"Node {node.id} has name {node.name}, expected {self.node_name}"
                )
            return found


class ArbitraryStrKeyAttributeMapper(Generic[AncestorT]):
    def __init__(
        self, stream: Iterable[AncestorT], keyfunc: Callable[[AncestorT], str]
    ):
        self.stream = stream
        self.keyfunc = keyfunc
        self.lookup_table = {keyfunc(_): _ for _ in stream}

    def __call__(self, node: Node) -> AncestorT:
        found = self.lookup_table[node.id]  # type: ignore
        return found


class NodeLookup(Mapping[Node, AdmitsNodeT]):
    def __init__(self, stream: Iterable[AdmitsNodeT]):
        self.stream = stream
        self.lookup_table = {_.as_node(): _ for _ in stream}

    def __getitem__(self, node: Node) -> AdmitsNodeT:
        return self.lookup_table[node]

    def __len__(self) -> int:
        return len(self.lookup_table)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.lookup_table)

    def __call__(self, node: Node) -> AdmitsNodeT:
        return self[node]


AnotherAdmitsNodeT = TypeVar("AnotherAdmitsNodeT", bound=AdmitsNode)


def get_children_of(
    query_nodes: Iterable[AdmitsNodeT],
    nodes_to_search: Iterable[AnotherAdmitsNodeT],
) -> Mapping[Node, Sequence[AnotherAdmitsNodeT]]:

    lookup = NodeLookup(query_nodes)

    parents_to_children: dict[Node, list[AnotherAdmitsNodeT]] = defaultdict(list)
    for node in nodes_to_search:
        # Check if the node to search admits any parent from the
        # query node.
        try:
            parent = get_ancestor_throw_on_failure(node.as_node(), lookup)
        except NoAncestorFound:
            continue
        else:
            parents_to_children[parent.as_node()].append(node)

    # Turn this from a defaultdict to a regular dict.
    return dict(parents_to_children)


def get_children_of_flattened(
    query_nodes: Iterable[AdmitsNodeT],
    nodes_to_search: Iterable[AnotherAdmitsNodeT],
) -> Sequence[AnotherAdmitsNodeT]:
    parents_to_children = get_children_of(query_nodes, nodes_to_search)
    return list(chain.from_iterable(parents_to_children.values()))


class DelayedSource(Mapping[Node, T]):
    """
    A base class for lazily loading data sources that map Node objects to values of type T.

    This class implements the Mapping interface, allowing it to be used like a dictionary.
    The actual data is loaded only when first accessed, which is useful for expensive
    data loading operations.

    Usage:
        1. Subclass DelayedSource and override __init__ and load_source methods
        2. In __init__, call super().__init__() and set up any parameters needed for loading
        3. In load_source, return a dictionary mapping Node objects to your data

    Example:
        ```python
        class TaskInstanceSource(DelayedSource[MathTaskInstance]):
            def __init__(self, task: MATHTask):
                super().__init__()
                self.task = task

            def load_source(self) -> dict[Node, MathTaskInstance]:
                return {
                    Node(
                        id=instance.instance_id,
                        name="originating_task_instance",
                        parent=None
                    ): instance
                    for instance in self.task.task_instances
                }

        # Create the source
        task = MATHTask(split="test_balanced_subset_10")
        source = TaskInstanceSource(task)

        # Use it like a dictionary - data is loaded only when first accessed
        for node, instance in source.items():
            print(f"Processing {instance.instance_id}")

        # Check if a node exists
        if some_node in source:
            # Get the corresponding value
            instance = source[some_node]
        ```

    The source data is loaded only once and cached for subsequent accesses.
    """

    def __init__(self):
        """
        Initialize the DelayedSource with an empty source.
        Subclasses should call super().__init__() and set up any parameters needed for loading.
        """
        self._source: Optional[dict[Node, T]] = None

    def load_source(self) -> dict[Node, T]:
        """
        Load and return the source data as a dictionary mapping Node objects to values.

        This method must be overridden by subclasses to define how to load the data.
        It will be called only once when the data is first accessed.

        Returns:
            dict[Node, T]: A dictionary mapping Node objects to values of type T

        Raises:
            NotImplementedError: If not overridden by a subclass
        """
        raise NotImplementedError("Subclasses must implement load_source()")

    @property
    def source(self) -> dict[Node, T]:
        """
        Property that lazily loads the source data when accessed.

        Returns:
            dict[Node, T]: The loaded source data
        """
        if self._source is None:
            self._source = self.load_source()
        return self._source

    def __getitem__(self, node: Node) -> T:
        return self.source[node]

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.source)

    def get(self, node: Node, default: Optional[T] = None) -> Optional[T]:
        return self.source.get(node, default)

    def refresh(self) -> None:
        """
        Force a reload of the source data on next access.

        This can be useful if the underlying data has changed.
        """
        self._source = None

    def source_mapper(self) -> NodeMapper[T]:
        """
        Get a NodeMapper that maps Nodes to the values in the source.

        Returns:
            NodeMapper[T]: A NodeMapper that maps Nodes to the values in the source
        """
        raise NotImplementedError("Subclasses must implement source_mapper()")
