"""
ALAKAZAM HDF5 I/O.

Unified format: always 5D (n_sol_time, n_sol_freq, n_ant, 2, 2).
Multi-SPW stored as /{jones_type}/spw_{id}/jones.
Complete metadata at every level.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger("alakazam")

FORMAT_VERSION = "2.0"


def save_solutions(
    filename: str,
    solutions: Dict[str, Dict[int, Dict[str, Any]]],
    observation: Dict[str, Any],
    config_yaml: str = "",
    overwrite: bool = True,
) -> None:
    """Save calibration solutions to HDF5.

    Parameters
    ----------
    filename : str
        Output HDF5 file path.
    solutions : dict
        Structure: {jones_type: {spw_id: {
            'jones': ndarray (n_t, n_f, n_ant, 2, 2),
            'time': ndarray (n_t,),
            'freq': ndarray (n_f,),  — chunk centers
            'freq_full': ndarray (n_chan,),  — all channels in SPW
            'flags': ndarray (n_t, n_f, n_ant),
            'weights': ndarray (n_t, n_f),
            'errors': ndarray or None,
            'params': dict of ndarray,  — native params
            'quality': dict of ndarray,  — per-cell metrics
            'metadata': dict,  — solver settings
        }}}
    observation : dict
        MS-level metadata (ms_path, antenna_names, working_ants, etc.)
    config_yaml : str
        Full YAML config string.
    overwrite : bool
        Overwrite existing file.
    """
    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise FileExistsError(f"File exists: {filename}. Use overwrite=True.")

    logger.info(f"Saving solutions to {filename}")

    with h5py.File(filename, "w") as f:
        # --- Per Jones type ---
        for jones_type, spw_data in solutions.items():
            jt_grp = f.create_group(jones_type)

            # Per-SPW data
            for spw_id, sol in spw_data.items():
                spw_grp = jt_grp.create_group(f"spw_{spw_id}")

                # Jones matrices — always 5D
                jones = np.asarray(sol["jones"], dtype=np.complex128)
                _ensure_5d(jones)
                spw_grp.create_dataset("jones", data=jones, compression="gzip")
                spw_grp["jones"].attrs["shape"] = "(n_sol_time, n_sol_freq, n_ant, 2, 2)"

                # Time and freq arrays
                spw_grp.create_dataset("time", data=np.asarray(sol["time"], dtype=np.float64))
                spw_grp.create_dataset("freq", data=np.asarray(sol["freq"], dtype=np.float64))
                if "freq_full" in sol:
                    spw_grp.create_dataset("freq_full", data=np.asarray(sol["freq_full"], dtype=np.float64))

                # Antenna array
                n_ant = jones.shape[2]
                spw_grp.create_dataset("antenna", data=np.arange(n_ant, dtype=np.int32))

                # Flags
                if "flags" in sol and sol["flags"] is not None:
                    spw_grp.create_dataset("flags", data=np.asarray(sol["flags"], dtype=bool), compression="gzip")

                # Weights
                if "weights" in sol and sol["weights"] is not None:
                    spw_grp.create_dataset("weights", data=np.asarray(sol["weights"], dtype=np.float64), compression="gzip")

                # Errors
                if "errors" in sol and sol["errors"] is not None:
                    spw_grp.create_dataset("errors", data=np.asarray(sol["errors"], dtype=np.float64), compression="gzip")

                # Native params
                if "params" in sol and sol["params"]:
                    pg = spw_grp.create_group("params")
                    for key, val in sol["params"].items():
                        if val is not None:
                            pg.create_dataset(key, data=np.asarray(val), compression="gzip")

                # Quality metrics
                if "quality" in sol and sol["quality"]:
                    qg = spw_grp.create_group("quality")
                    for key, val in sol["quality"].items():
                        if val is not None:
                            qg.create_dataset(key, data=np.asarray(val))

            # Jones-level metadata (shared across SPWs)
            meta_grp = jt_grp.create_group("metadata")
            first_sol = next(iter(spw_data.values()))
            if "metadata" in first_sol:
                for key, val in first_sol["metadata"].items():
                    if val is None:
                        continue
                    if isinstance(val, (str, int, float, bool, np.integer, np.floating)):
                        meta_grp.attrs[str(key)] = val
                    elif isinstance(val, np.ndarray):
                        meta_grp.create_dataset(str(key), data=val)

        # --- Observation metadata ---
        obs_grp = f.create_group("observation")
        for key, val in observation.items():
            if val is None:
                continue
            if isinstance(val, (str, int, float, bool, np.integer, np.floating)):
                obs_grp.attrs[str(key)] = val
            elif isinstance(val, np.ndarray):
                obs_grp.create_dataset(str(key), data=val)
            elif isinstance(val, list):
                if all(isinstance(v, str) for v in val):
                    obs_grp.create_dataset(str(key), data=np.array(val, dtype="S"))
                else:
                    obs_grp.create_dataset(str(key), data=np.array(val))

        # --- Config ---
        cfg_grp = f.create_group("config")
        cfg_grp.attrs["yaml_content"] = config_yaml
        cfg_grp.attrs["solve_order"] = ",".join(solutions.keys())

        # --- File metadata ---
        f.attrs["creation_time"] = datetime.now().isoformat()
        f.attrs["alakazam_version"] = "2.0.0"
        f.attrs["format_version"] = FORMAT_VERSION

    logger.info(f"Saved {len(solutions)} Jones types to {filename}")


def load_solutions(
    filename: str,
    jones_types: Optional[List[str]] = None,
    spw_ids: Optional[List[int]] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load calibration solutions from HDF5.

    Returns: {jones_type: {spw_id: {'jones': ..., 'time': ..., ...}}}
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    result = {}
    reserved = {"observation", "config"}

    with h5py.File(filename, "r") as f:
        available_jt = [k for k in f.keys() if k not in reserved and isinstance(f[k], h5py.Group)]

        if jones_types is None:
            jones_types = available_jt

        for jt in jones_types:
            if jt not in f:
                logger.warning(f"Jones type '{jt}' not found in {filename}")
                continue

            jt_grp = f[jt]
            result[jt] = {}

            spw_keys = [k for k in jt_grp.keys() if k.startswith("spw_")]
            for spw_key in spw_keys:
                sid = int(spw_key.split("_")[1])
                if spw_ids is not None and sid not in spw_ids:
                    continue

                sg = jt_grp[spw_key]
                sol = {
                    "jones": sg["jones"][:],
                    "time": sg["time"][:],
                    "freq": sg["freq"][:],
                }

                if "freq_full" in sg:
                    sol["freq_full"] = sg["freq_full"][:]
                if "flags" in sg:
                    sol["flags"] = sg["flags"][:]
                if "weights" in sg:
                    sol["weights"] = sg["weights"][:]
                if "errors" in sg:
                    sol["errors"] = sg["errors"][:]
                if "antenna" in sg:
                    sol["antenna"] = sg["antenna"][:]

                # Params
                sol["params"] = {}
                if "params" in sg:
                    for pk in sg["params"]:
                        sol["params"][pk] = sg["params"][pk][:]

                # Quality
                sol["quality"] = {}
                if "quality" in sg:
                    for qk in sg["quality"]:
                        sol["quality"][qk] = sg["quality"][qk][:]

                result[jt][sid] = sol

    return result


def print_summary(filename: str) -> str:
    """Print summary of solution file. Returns summary string."""
    filename = Path(filename)
    lines = [f"\nALAKAZAM Solution: {filename}", "=" * 60]

    with h5py.File(filename, "r") as f:
        lines.append(f"Version: {f.attrs.get('alakazam_version', '?')}")
        lines.append(f"Created: {f.attrs.get('creation_time', '?')}")

        if "observation" in f:
            obs = f["observation"]
            lines.append(f"\nObservation:")
            for k in obs.attrs:
                lines.append(f"  {k}: {obs.attrs[k]}")

        reserved = {"observation", "config"}
        jt_list = [k for k in f.keys() if k not in reserved]
        lines.append(f"\nJones types: {', '.join(jt_list)}")

        for jt in jt_list:
            if not isinstance(f[jt], h5py.Group):
                continue
            lines.append(f"\n{jt}:")
            spw_keys = sorted(k for k in f[jt].keys() if k.startswith("spw_"))
            for sk in spw_keys:
                sg = f[jt][sk]
                if "jones" in sg:
                    shape = sg["jones"].shape
                    lines.append(f"  {sk}: jones {shape}")
                    if "quality" in sg and "chi2_red" in sg["quality"]:
                        chi2 = sg["quality/chi2_red"][:]
                        valid = chi2[np.isfinite(chi2)]
                        if len(valid) > 0:
                            lines.append(f"    chi2_red: median={np.median(valid):.3f}")

    summary = "\n".join(lines) + "\n"
    return summary


def _ensure_5d(jones: np.ndarray):
    """Validate Jones array is 5D."""
    if jones.ndim != 5:
        raise ValueError(
            f"Jones array must be 5D (n_t, n_f, n_ant, 2, 2), got shape {jones.shape}"
        )
    if jones.shape[3] != 2 or jones.shape[4] != 2:
        raise ValueError(
            f"Jones array last two dims must be (2,2), got {jones.shape}"
        )
