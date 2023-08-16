from __future__ import annotations

import threading
from typing import TYPE_CHECKING, NamedTuple

from tensorboard.backend.event_processing import (
    directory_watcher,
    event_file_loader,
    io_wrapper,
    reservoir,
)

if TYPE_CHECKING:
    from tensorboard.compat.proto.event_pb2 import Event


class ScalarEvent(NamedTuple):
    """A logged scalar value."""

    wall_time: float
    step: int
    value: float


class EventAccumulator:
    """Stripped-down version of TensorBoard's EventAccumulator that Reloads() only
    scalars. Speeds up the loading process for large event files with e.g. histograms by
    about 4x.

    Args:
        path: A file path to a directory containing tf events files, or a single
        tf events file. The accumulator will load events from this path.

    The EventAccumulator is intended to provide a convenient Python interface
    for loading Event data written during a TensorFlow run. TensorFlow writes out
    Event protobuf objects, which have a timestamp and step number, and often
    contain a Summary. Summaries can have different kinds of data like an image,
    a scalar value, or a histogram. The Summaries also have a tag, which we use to
    organize logically related data. The EventAccumulator supports retrieving
    the Event and Summary data by its tag.

    Calling Tags() gets a map from tagType (e.g. 'images', 'scalars', etc) to
    the associated tags for those data types. Then, various functional endpoints
    (e.g. Accumulator.Scalars(tag)) allow for the retrieval of all data
    associated with that tag.

    Fields:
        path: A file path to a directory containing tf events files, or a single
            tf events file. The accumulator will load events from this path.
        scalars: A reservoir.Reservoir of scalar summaries.
    """

    def __init__(self, path: str) -> None:
        """Create a new EventAccumulator which is a generator that yields Event objects
        as well as a Reservoir object to store the last 10,000 Events.

        Args:
            path (str): The path to the event file.
        """
        self._first_event_timestamp = None
        self.scalars = reservoir.Reservoir(size=10000)

        self._generator_mutex = threading.Lock()
        self.path = path
        self._generator = _GeneratorFromPath(path)

        self.file_version: float | None = None

    def Reload(self) -> EventAccumulator:
        """Synchronously load all events added since last calling Reload. If Reload was
        never called, loads all events in the file.

        Returns:
            EventAccumulator
        """
        with self._generator_mutex:
            for event in self._generator.Load():
                self._ProcessEvent(event)
        return self

    def _ProcessEvent(self, event: Event) -> None:
        """Called whenever an event is loaded."""
        if self._first_event_timestamp is None:
            self._first_event_timestamp = event.wall_time

        if event.HasField("file_version"):
            new_file_version = _ParseFileVersion(event.file_version)
            self.file_version = new_file_version

        if event.HasField("summary"):
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    datum = value.simple_value
                    tag = value.tag
                    self._ProcessScalar(tag, event.wall_time, event.step, datum)

    @property
    def scalar_tags(self) -> list[str]:
        """Return all scalar tags found in the value stream.

        Returns:
            list[str]: All scalar tags
        """
        return self.scalars.Keys()

    def Scalars(self, tag: str) -> tuple[ScalarEvent, ...]:
        """Given a summary tag, return all associated ScalarEvents.

        Args:
            tag (str): The tag associated with the desired events.

        Raises:
            KeyError: If the tag is not found.

        Returns:
            tuple[ScalarEvent, ...]: An array of ScalarEvents.
        """
        return self.scalars.Items(tag)

    def _ProcessScalar(
        self, tag: str, wall_time: float, step: int, scalar: float
    ) -> None:
        """Process a simple value by adding it to accumulated state."""
        sv = ScalarEvent(wall_time=wall_time, step=step, value=scalar)
        self.scalars.AddItem(tag, sv)


def _GeneratorFromPath(path: str) -> directory_watcher.DirectoryWatcher:
    """Create an event generator for file or directory at given path string."""
    return directory_watcher.DirectoryWatcher(
        path,
        event_file_loader.LegacyEventFileLoader,
        io_wrapper.IsSummaryEventsFile,
    )


def _ParseFileVersion(file_version: str) -> float:
    """Convert the string file_version in event.proto into a float.

    Args:
      file_version: String file_version from event.proto

    Returns:
      Version number as a float.
    """
    tokens = file_version.split("brain.Event:")
    return float(tokens[-1])
