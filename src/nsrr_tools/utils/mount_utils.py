"""
SSHFS mount health-check and auto-remount utility.

Compute Canada requires OTP on every NEW SSH connection, so fully passwordless
remounting is only possible when an SSH ControlMaster socket already exists
(i.e. you authenticated once in a terminal today).

Setup (one-time):
    Add to ~/.ssh/config:
        Host fir.alliancecan.ca
            User boshra95
            ControlMaster auto
            ControlPath ~/.ssh/cm-%r@%h:%p
            ControlPersist 10h

Daily workflow:
    1. In any terminal: ssh fir.alliancecan.ca  (enter password + OTP once)
    2. Keep that terminal open or just close it — ControlPersist keeps the socket alive.
    3. sshfs and this utility will reuse the socket silently for ~10 hours.

Usage in any extraction script's main():

    from nsrr_tools.utils.mount_utils import ensure_sshfs_mounted

    ensure_sshfs_mounted(
        mount_point=Path(config['paths']['raw_data']).parent,
        remote="boshra95@fir.alliancecan.ca:/home/boshra95/scratch/",
        options=["auto_cache", "reconnect", "compression=yes"],
    )
"""

import errno
import os
import subprocess
import time
from pathlib import Path

from loguru import logger


def _is_mounted(path: Path) -> bool:
    """Return True if path is accessible (not a stale mount)."""
    try:
        os.stat(path)
        return True
    except OSError as e:
        if e.errno in (errno.ENODEV, errno.ENXIO, errno.EIO):
            # errno 6  = ENXIO  (Device not configured) — stale SSHFS on macOS
            # errno 19 = ENODEV (No such device)
            # errno 5  = EIO    (I/O error)
            return False
        raise  # unexpected error — let it propagate


def _force_umount(mount_point: Path) -> None:
    """Unmount a stale SSHFS mount. Tries umount, then diskutil on macOS."""
    mp = str(mount_point)
    logger.info(f"Unmounting stale mount: {mp}")

    result = subprocess.run(["umount", mp], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("umount succeeded")
        return

    # Fallback: diskutil (macOS)
    result2 = subprocess.run(
        ["diskutil", "unmount", "force", mp],
        capture_output=True, text=True,
    )
    if result2.returncode == 0:
        logger.info("diskutil unmount force succeeded")
        return

    # Neither worked — log and continue anyway (sshfs may still remount)
    logger.warning(f"umount failed: {result.stderr.strip()} | diskutil: {result2.stderr.strip()}")


def _control_socket_exists(host: str, user: str) -> bool:
    """Check if a live SSH ControlMaster socket exists for this host/user."""
    control_path = Path.home() / ".ssh" / f"cm-{user}@{host}:22"
    if not control_path.exists():
        return False
    # Confirm the socket is actually alive (not a stale file)
    result = subprocess.run(
        ["ssh", "-O", "check", "-o", f"ControlPath={control_path}", f"{user}@{host}"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def _do_mount(mount_point: Path, remote: str, options: list[str]) -> bool:
    """
    Run sshfs to mount remote → mount_point.
    Injects ControlPath so sshfs reuses an existing authenticated SSH socket
    (required for Compute Canada OTP — no interactive prompt possible).
    Returns True if sshfs exits successfully (it daemonises, so exit 0 = mounted).
    """
    mp = str(mount_point)

    # Extract user@host from remote string (e.g. "boshra95@fir.alliancecan.ca:/path/")
    remote_part = remote.split(":")[0]  # "boshra95@fir.alliancecan.ca"
    user, host = remote_part.split("@") if "@" in remote_part else ("", remote_part)
    control_path = str(Path.home() / ".ssh" / f"cm-{user}@{host}:22")

    # Always pass ControlPath so sshfs piggybacks on an open session
    ssh_opts = f"ControlPath={control_path}"
    all_options = options + [f"ssh_command=ssh -o {ssh_opts}"]

    cmd = ["sshfs", remote, mp, "-o", ",".join(all_options)]

    logger.info(f"Mounting via ControlPath {control_path}")
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        logger.info("sshfs mounted successfully")
        return True

    logger.error(f"sshfs failed (rc={result.returncode}): {result.stderr.strip()}")
    return False


def ensure_sshfs_mounted(
    mount_point: Path,
    remote: str,
    options: list[str] | None = None,
    retry_wait: float = 2.0,
) -> None:
    """
    Check that mount_point is accessible. If not, unmount and remount via sshfs.

    Args:
        mount_point: Local directory that should be the SSHFS mount root.
        remote:      SSHFS remote string, e.g. "user@host:/remote/path/".
        options:     List of -o options (without the -o flag), e.g.
                     ["auto_cache", "reconnect", "compression=yes"].
        retry_wait:  Seconds to wait after mounting before re-checking accessibility.

    Raises:
        RuntimeError: If the mount cannot be restored.
    """
    options = options or []
    mount_point = Path(mount_point)

    if _is_mounted(mount_point):
        logger.debug(f"Mount OK: {mount_point}")
        return

    logger.warning(f"Stale or disconnected mount detected: {mount_point}")
    logger.warning("Attempting automatic remount…")

    _force_umount(mount_point)
    mount_point.mkdir(parents=True, exist_ok=True)

    ok = _do_mount(mount_point, remote, options)
    if not ok:
        raise RuntimeError(
            f"Could not remount {remote} → {mount_point}.\n\n"
            "Compute Canada requires OTP. To enable automatic remounting:\n"
            "  1. Add ControlMaster to ~/.ssh/config (see mount_utils.py docstring)\n"
            "  2. Open a terminal and run:  ssh fir.alliancecan.ca  (enter OTP once)\n"
            "  3. Re-run this script — it will reuse the open session.\n\n"
            "Or mount manually:\n"
            f"  sshfs {remote} {mount_point} -o {','.join(options)}"
        )

    # Give the FUSE daemon a moment to finish populating the mount
    time.sleep(retry_wait)

    if not _is_mounted(mount_point):
        raise RuntimeError(
            f"sshfs reported success but {mount_point} is still inaccessible. "
            "Check SSH agent / key forwarding."
        )

    logger.info(f"Remount successful: {mount_point}")
