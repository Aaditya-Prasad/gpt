from __future__ import annotations

"""Simple data interface to work with generated prompts, tokens and (optionally)
activations produced by the generation + pre-fill pipeline.

Directory layout expected::

    <output_dir>/
        tokens/                         # torch tensors containing generated tokens
        data.json                       # metadata such as partition_size & num_prompts
        config.yaml                     # generation configuration (unused here)
        prompts_and_completions.json    # list[{prompt, completion, conversation_id}]
        activations/                    # optional – added later by prefill.py

The *tokens/* directory contains a number of torch *.pt files (or *.pth).  Each file
holds ``partition_size`` rows, where *partition_size* is saved in *data.json*.
The first dimension (dim-0) of every tensor corresponds to prompts **within** the
partition, therefore the row offset for a global ``uid`` is::

    partition_idx = uid // partition_size
    row_idx       = uid %  partition_size

This class maps every *conversation_id* (string) to a deterministic integer *uid*
(index in *prompts_and_completions.json*) so that callers can reference examples
using either identifier.

Notes
-----
* "activations" loading is **not implemented yet** – a ``NotImplementedError`` is
  raised when requested.  The remaining interface is forward-compatible so that
  the logic can be added later.
* ``**kwargs`` / ``*args`` are intentionally avoided per user guidelines.
"""

from pathlib import Path
import json
from typing import Dict, List, Sequence, Union, Tuple, Set

import numpy as np

__all__ = ["DataInterface"]


_Row = Dict[str, str]  # shorthand type alias for one entry in prompts_and_completions
LoadType = Union[str, Sequence[str]]
Identifier = Union[int, str]


class DataInterface:  # pylint: disable=too-few-public-methods
    """Helper for reading prompts, tokens and (future) activations from disk."""

    # ------------------------- public API --------------------------------- #

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        if not self.output_dir.exists():
            raise FileNotFoundError(f"{self.output_dir} does not exist")

        # Mandatory paths
        self.tokens_dir = self._require_path("tokens", is_dir=True)
        self.data_json_path = self._require_path("data.json")
        self.prompts_json_path = self._require_path("prompts_and_completions.json")
        # config.yaml is optional for this interface, do not fail if missing

        # Optional – might be absent during initial generation
        self.activations_dir = self.output_dir / "activations"
        self._has_activations = self.activations_dir.exists()

        # Load metadata
        with open(self.data_json_path, "r", encoding="utf-8") as fh:
            data_meta = json.load(fh)
        self._data: Dict[str, object] = data_meta  # store full metadata for easy access

        try:
            self.partition_size: int = int(data_meta["partition_size"])
        except (KeyError, ValueError) as exc:
            raise KeyError("'partition_size' must exist and be an int in data.json") from exc

        # Prompts & completions
        with open(self.prompts_json_path, "r", encoding="utf-8") as fh:
            self._prompts: List[_Row] = json.load(fh)

        if not isinstance(self._prompts, list):
            raise ValueError("prompts_and_completions.json should contain a list")

        self.num_rows: int = len(self._prompts)

        # Deterministic uid \u2194 conversation_id mapping
        self._uid_to_cid: List[str] = []  # index -> cid
        self._cid_to_uid: Dict[str, int] = {}

        for idx, row in enumerate(self._prompts):
            cid = row.get("conversation_id")
            if cid is None:
                raise ValueError(f"Row {idx} is missing 'conversation_id'")
            if cid in self._cid_to_uid:
                raise ValueError(f"Duplicate conversation_id encountered: {cid}")
            self._uid_to_cid.append(cid)
            self._cid_to_uid[cid] = idx

        # Cache for already loaded token partitions: {partition_idx: torch.Tensor}
        self._token_partition_cache: Dict[int, np.ndarray] = {}

        # Sorted list of token file paths for deterministic indexing
        self._token_files: List[Path] = self._discover_token_files()
        if not self._token_files:
            raise FileNotFoundError("No token partition files found under tokens/ directory")

        # Quick health check to catch glaring issues early (optional heavy).
        self._performed_quick_health_check = False

    # --------------------------------------------------------------------- #

    def check_health(self) -> bool:
        """Run a batch of consistency checks on the dataset.

        * Each row must contain *prompt*, *completion* and *conversation_id* keys.
        * Number of sequences across *tokens/* partitions must equal number of rows.

        Returns
        -------
        bool
            ``True`` if all checks pass, otherwise an exception is raised.
        """
        # Validate json rows
        required_keys: Set[str] = {"prompt", "completion", "conversation_id"}
        for idx, row in enumerate(self._prompts):
            missing = required_keys.difference(row)
            if missing:
                raise ValueError(f"Row {idx} missing keys: {', '.join(missing)}")

        # Count sequences in token partitions
        total_sequences = 0
        for pth in self._token_files:
            tensor = self._load_token_partition(pth)
            total_sequences += tensor.shape[0]

        if total_sequences != self.num_rows:
            raise ValueError(
                "Mismatch between number of sequences in tokens/ "
                f"({total_sequences}) and number of rows in prompts JSON ({self.num_rows})"
            )

        self._performed_quick_health_check = True
        return True

    # --------------------------------------------------------------------- #

    def load(
        self,
        *,
        uids: int | Sequence[int] | None = None,
        conversation_ids: str | Sequence[str] | None = None,
        load_type: LoadType = "text",
    ) -> Dict[str, object]:
        """Retrieve information for given *uid*/*conversation_id* identifiers.

        Parameters
        ----------
        uids / conversation_ids
            Provide exactly one: *uids* (``int`` or sequence of ``int``) **or** *conversation_ids* (``str`` or sequence of ``str``).
        load_type
            What to load. Options:
                * ``"text"`` – return dict(s) with *prompt*, *completion*, *conversation_id*
                * ``"tokens"`` – return corresponding token tensor(s)
                * ``"activations"`` – *Not implemented yet*
                * ``"all"`` – return dict(s) that include whichever of the above are available
            ``load_type`` can also be a list/tuple of the individual options (**excluding** "all").

        Notes
        -----
        * If *activations* are requested but not yet implemented, ``NotImplementedError`` is raised.
        * For a single identifier the values are returned directly (not wrapped in a list).
        * A dictionary is always returned; keys correspond to requested data types.
        """
        # Validate identifier inputs
        if (uids is None) == (conversation_ids is None):
            raise ValueError("Provide exactly one of 'uids' or 'conversation_ids'.")

        if uids is not None:
            uid_list, is_single = self._normalize_uids(uids)
        else:
            uid_list, is_single = self._normalize_cids(conversation_ids)  # type: ignore[arg-type]

        load_type_set = self._normalize_load_type(load_type)

        # Gather per-uid dictionaries
        per_uid_results = [self._load_single(uid, load_type_set) for uid in uid_list]

        # Aggregate into dict of lists/objects
        aggregate: Dict[str, List[object]] = {}
        for res in per_uid_results:
            for key, value in res.items():
                aggregate.setdefault(key, []).append(value)

        if is_single:
            # unwrap single-element lists for convenience
            for key in aggregate:
                aggregate[key] = aggregate[key][0]  # type: ignore[index]
        return aggregate

    # ------------------------- private helpers --------------------------- #

    # Path helpers ------------------------------------------------------- #

    def _require_path(self, relative: str, *, is_dir: bool = False) -> Path:
        pth = self.output_dir / relative
        if is_dir and not pth.is_dir():
            raise FileNotFoundError(f"Required directory '{relative}' missing under {self.output_dir}")
        if not is_dir and not pth.is_file():
            raise FileNotFoundError(f"Required file '{relative}' missing under {self.output_dir}")
        return pth

    def _discover_token_files(self) -> List[Path]:
        files = sorted(self.tokens_dir.glob("*.npy"))
        return files

    # Token loading ------------------------------------------------------ #

    def _get_partition_index(self, uid: int) -> Tuple[int, int]:
        partition_idx = uid // self.partition_size
        row_idx = uid % self.partition_size
        return partition_idx, row_idx

    def _load_token_partition(self, path: Path) -> np.ndarray:
        # Basic caching – avoids re-loading the same partition multiple times
        partition_idx = self._token_files.index(path)
        if partition_idx in self._token_partition_cache:
            return self._token_partition_cache[partition_idx]

        array = np.load(path, allow_pickle=False)
        if array.ndim < 2:
            raise ValueError(f"Token partition at {path} does not have expected 2+ dims")
        self._token_partition_cache[partition_idx] = array
        return array

    def _load_tokens_by_uid(self, uid: int) -> np.ndarray:
        partition_idx, row_idx = self._get_partition_index(uid)
        try:
            partition_path = self._token_files[partition_idx]
        except IndexError as exc:
            raise IndexError(
                f"Partition index {partition_idx} out of range for uid {uid}. "
                "Check partition_size and number of token files."
            ) from exc

        partition_tensor = self._load_token_partition(partition_path)
        if row_idx >= partition_tensor.shape[0]:
            raise IndexError(
                f"Row {row_idx} exceeds partition batch dimension {partition_tensor.shape[0]} "
                f"(uid={uid}, partition={partition_idx})."
            )
        return partition_tensor[row_idx]

    # Load routing ------------------------------------------------------- #

    def _load_single(self, uid: int, load_type_set: Set[str]):
        include_text = "text" in load_type_set or "all" in load_type_set
        include_tokens = "tokens" in load_type_set or "all" in load_type_set
        include_activations = "activations" in load_type_set or "all" in load_type_set

        if include_activations:
            raise NotImplementedError(
                "Loading activations is not yet supported – will be implemented later."
            )

        # Always start with text – shallow copy to avoid mutating cached list
        row = dict(self._prompts[uid]) if include_text else None
        tokens = self._load_tokens_by_uid(uid) if include_tokens else None

        # Always return a dictionary keyed by the requested data types
        bundle: Dict[str, object] = {}
        if include_text:
            bundle["text"] = row  # type: ignore[arg-type]
        if include_tokens:
            bundle["tokens"] = tokens  # type: ignore[arg-type]
        # Activations intentionally omitted for now

        return bundle

    # Normalisation utilities ------------------------------------------- #

    def _normalize_uids(self, uids: int | Sequence[int]) -> Tuple[List[int], bool]:
        """Validate and normalize *uids* into a list of int plus *is_single* flag."""
        if isinstance(uids, (list, tuple)):
            uid_list = list(uids)
            is_single = False
        else:
            uid_list = [uids]  # type: ignore[list-item]
            is_single = True

        for uid in uid_list:
            if not isinstance(uid, int):
                raise TypeError(f"UIDs must be integers; got {type(uid).__name__}")
        return uid_list, is_single

    def _normalize_cids(self, conversation_ids: str | Sequence[str]) -> Tuple[List[int], bool]:
        """Convert conversation_id(s) to uid list and flag if single."""
        if isinstance(conversation_ids, (list, tuple)):
            cid_list = list(conversation_ids)
            is_single = False
        else:
            cid_list = [conversation_ids]  # type: ignore[list-item]
            is_single = True

        uid_list: List[int] = []
        for cid in cid_list:
            if not isinstance(cid, str):
                raise TypeError(f"conversation_id must be str; got {type(cid).__name__}")
            try:
                uid_list.append(self._cid_to_uid[cid])
            except KeyError as exc:
                raise ValueError(f"Unknown conversation_id: {cid}") from exc
        return uid_list, is_single

    def _normalize_load_type(self, load_type: LoadType) -> Set[str]:
        valid = {"text", "tokens", "activations", "all"}
        if isinstance(load_type, str):
            if load_type not in valid:
                raise ValueError(f"load_type must be one of {valid}; got '{load_type}'")
            return {load_type}
        else:
            # sequence – ensure each entry is valid and not 'all' (ambiguous)
            load_set: Set[str] = set()
            for lt in load_type:
                if lt == "all":
                    raise ValueError("'all' cannot be combined with other load types")
                if lt not in valid:
                    raise ValueError(f"Invalid load_type entry: '{lt}'")
                load_set.add(lt)
            return load_set

    # -------------------------------------------------------------------- #

    # Optional convenience dunder methods -------------------------------- #

    def __len__(self) -> int:  # noqa: Dunder method
        return self.num_rows

    # Explicit method alternative to the builtin len(ds)
    def len(self) -> int:
        """Return number of available rows (prompts)."""
        return self.num_rows

    # -------------------------------------------------------------------- #

    def data(self, key: str):
        """Return value for *key* from ``data.json``.

        Parameters
        ----------
        key
            The key to look up in *data.json*.
        Raises
        ------
        KeyError
            If the key is not present.
        """
        try:
            return self._data[key]
        except KeyError as exc:
            raise KeyError(
                f"'{key}' not found in data.json; available keys: {list(self._data.keys())}"
            ) from exc

    def __getitem__(self, item: int):  # noqa: Dunder method
        return self.load(uids=item, load_type="text")

    # -------------------------------------------------------------------- #

    def __repr__(self) -> str:  # noqa: Dunder method
        return (
            f"DataInterface(n_rows={self.num_rows}, partitions={len(self._token_files)}, "
            f"partition_size={self.partition_size}, activations={self._has_activations})"
        )

