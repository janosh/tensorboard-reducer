import threading
from collections import namedtuple

from tensorboard.backend.event_processing import (
    directory_watcher,
    event_file_loader,
    io_wrapper,
    reservoir,
)


ScalarEvent = namedtuple("ScalarEvent", ["wall_time", "step", "value"])


class EventAccumulator:
    """Stripped-down version of TensorBoard's EventAccumulator that Reloads()
    only scalars. Speeds up the loading process for large event files with e.g.
    histograms by about 4x.

    An `EventAccumulator` takes an event generator, and accumulates the
    values.

    The `EventAccumulator` is intended to provide a convenient Python interface
    for loading Event data written during a TensorFlow run. TensorFlow writes out
    `Event` protobuf objects, which have a timestamp and step number, and often
    contain a `Summary`. Summaries can have different kinds of data like an image,
    a scalar value, or a histogram. The Summaries also have a tag, which we use to
    organize logically related data. The `EventAccumulator` supports retrieving
    the `Event` and `Summary` data by its tag.

    Calling `Tags()` gets a map from `tagType` (e.g. 'images', 'scalars', etc) to
    the associated tags for those data types. Then, various functional endpoints (eg
    `Accumulator.Scalars(tag)`) allow for the retrieval of all data
    associated with that tag.

    The `Reload()` method synchronously loads all scalar data written so far.

    Fields:
      most_recent_step: Step of last Event proto added. This should only
          be accessed from the thread that calls Reload. This is -1 if
          nothing has been loaded yet.
      most_recent_wall_time: Timestamp of last Event proto added. This is
          a float containing seconds from the UNIX epoch, or -1 if
          nothing has been loaded yet. This should only be accessed from
          the thread that calls Reload.
      path: A file path to a directory containing tf events files, or a single
          tf events file. The accumulator will load events from this path.
      scalars: A reservoir.Reservoir of scalar summaries.
    """

    def __init__(self, path: str) -> None:
        """Construct the `EventAccumulator`.

        Args:
          path: A file path to a directory containing tf events files, or a single
            tf events file. The accumulator will load events from this path.
          size_guidance: Information on how much data the EventAccumulator should
            store in memory. The DEFAULT_SIZE_GUIDANCE tries not to store too much
            so as to avoid OOMing the client. The size_guidance should be a map
            from a `tagType` string to an integer representing the number of
            items to keep per tag for items of that `tagType`. If the size is 0,
            all events are stored.
        """

        self._first_event_timestamp = None
        self.scalars = reservoir.Reservoir(size=10000)

        self._tagged_metadata = {}
        self.summary_metadata = {}

        self._generator_mutex = threading.Lock()
        self.path = path
        self._generator = _GeneratorFromPath(path)

        self.most_recent_step = -1
        self.most_recent_wall_time = -1
        self.file_version = None

        # The attributes that get built up by the accumulator
        self.accumulated_attrs = ("scalars",)
        self._tensor_summaries = {}

    def Reload(self) -> "EventAccumulator":
        """Loads all events added since the last call to `Reload`.

        If `Reload` was never called, loads all events in the file.

        Returns:
          The `EventAccumulator`.
        """
        with self._generator_mutex:
            for event in self._generator.Load():
                self._ProcessEvent(event)
        return self

    def _ProcessEvent(self, event):
        """Called whenever an event is loaded."""
        if self._first_event_timestamp is None:
            self._first_event_timestamp = event.wall_time

        if event.HasField("file_version"):
            new_file_version = _ParseFileVersion(event.file_version)
            if self.file_version and self.file_version != new_file_version:
                # This should not happen.
                print(
                    (
                        "Found new file_version for event.proto. This will "
                        "affect purging logic for TensorFlow restarts. "
                        "Old: {} New: {}"
                    ).format(self.file_version, new_file_version)
                )
            self.file_version = new_file_version

        if event.HasField("summary"):
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    datum = getattr(value, "simple_value")
                    tag = value.tag
                    if "simple_value" == "tensor" and not tag:
                        # This tensor summary was created using the old method that used
                        # plugin assets. We must still continue to support it.
                        tag = value.node_name
                    self._ProcessScalar(tag, event.wall_time, event.step, datum)

    def Tags(self):
        """Return all tags found in the value stream.

        Returns:
          A `{tagType: ['list', 'of', 'tags']}` dictionary.
        """
        return {"scalars": self.scalars.Keys()}

    def Scalars(self, tag):
        """Given a summary tag, return all associated `ScalarEvent`s.

        Args:
          tag: A string tag associated with the events.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          An array of `ScalarEvent`s.
        """
        return self.scalars.Items(tag)

    def _ProcessScalar(self, tag, wall_time, step, scalar):
        """Processes a simple value by adding it to accumulated state."""
        sv = ScalarEvent(wall_time=wall_time, step=step, value=scalar)
        self.scalars.AddItem(tag, sv)


def _GeneratorFromPath(path):
    """Create an event generator for file or directory at given path string."""
    if not path:
        raise ValueError("path must be a valid string")
    if io_wrapper.IsSummaryEventsFile(path):
        return event_file_loader.LegacyEventFileLoader(path)
    else:
        return directory_watcher.DirectoryWatcher(
            path,
            event_file_loader.LegacyEventFileLoader,
            io_wrapper.IsSummaryEventsFile,
        )


def _ParseFileVersion(file_version):
    """Convert the string file_version in event.proto into a float.

    Args:
      file_version: String file_version from event.proto

    Returns:
      Version number as a float.
    """
    tokens = file_version.split("brain.Event:")
    try:
        return float(tokens[-1])
    except ValueError:
        # This should never happen according to the definition of file_version
        # specified in event.proto.
        print(
            "Invalid event.proto file_version. Defaulting to use of "
            "out-of-order event.step logic for purging expired events."
        )
        return -1
