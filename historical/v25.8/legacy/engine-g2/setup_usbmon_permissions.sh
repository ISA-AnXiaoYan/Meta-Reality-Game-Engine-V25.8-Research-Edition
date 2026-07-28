#!/usr/bin/env bash
set -euo pipefail

GROUP_NAME="${USBMON_GROUP:-usbmon}"
TARGET_USER="${SUDO_USER:-${USER:-}}"
SERVICE_FILE="/etc/systemd/system/ids-usbmon-permissions.service"

if [ "$(id -u)" -ne 0 ]; then
  echo "[usbmon][ERROR] please run with sudo:"
  echo "  sudo bash ./setup_usbmon_permissions.sh"
  exit 1
fi

if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = "root" ]; then
  echo "[usbmon][ERROR] cannot determine non-root target user. Set SUDO_USER or run via sudo from the launch user."
  exit 1
fi

echo "[usbmon] target_user=$TARGET_USER group=$GROUP_NAME"

modprobe usbmon || true

if ! mountpoint -q /sys/kernel/debug; then
  mount -t debugfs none /sys/kernel/debug
fi

if ! getent group "$GROUP_NAME" >/dev/null 2>&1; then
  groupadd "$GROUP_NAME"
fi

usermod -aG "$GROUP_NAME" "$TARGET_USER"

apply_now() {
  modprobe usbmon || true
  if ! mountpoint -q /sys/kernel/debug; then
    mount -t debugfs none /sys/kernel/debug || true
  fi
  if [ -d /sys/kernel/debug/usb/usbmon ]; then
    chgrp "$GROUP_NAME" /sys/kernel/debug /sys/kernel/debug/usb /sys/kernel/debug/usb/usbmon 2>/dev/null || true
    chmod g+x /sys/kernel/debug /sys/kernel/debug/usb 2>/dev/null || true
    chmod g+rx /sys/kernel/debug/usb/usbmon 2>/dev/null || true
    chgrp "$GROUP_NAME" /sys/kernel/debug/usb/usbmon/*u 2>/dev/null || true
    chmod g+r /sys/kernel/debug/usb/usbmon/*u 2>/dev/null || true
    if command -v setfacl >/dev/null 2>&1; then
      setfacl -m "u:${TARGET_USER}:x" /sys/kernel/debug /sys/kernel/debug/usb 2>/dev/null || true
      setfacl -m "u:${TARGET_USER}:rx" /sys/kernel/debug/usb/usbmon 2>/dev/null || true
      setfacl -m "u:${TARGET_USER}:r" /sys/kernel/debug/usb/usbmon/*u 2>/dev/null || true
    fi
  fi
}

apply_now

cat > "$SERVICE_FILE" <<UNIT
[Unit]
Description=Grant IDS usbmon read permissions
After=systemd-modules-load.service

[Service]
Type=oneshot
ExecStart=/bin/bash -lc 'modprobe usbmon || true; mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug || true; if [ -d /sys/kernel/debug/usb/usbmon ]; then chgrp ${GROUP_NAME} /sys/kernel/debug /sys/kernel/debug/usb /sys/kernel/debug/usb/usbmon 2>/dev/null || true; chmod g+x /sys/kernel/debug /sys/kernel/debug/usb 2>/dev/null || true; chmod g+rx /sys/kernel/debug/usb/usbmon 2>/dev/null || true; chgrp ${GROUP_NAME} /sys/kernel/debug/usb/usbmon/*u 2>/dev/null || true; chmod g+r /sys/kernel/debug/usb/usbmon/*u 2>/dev/null || true; if command -v setfacl >/dev/null 2>&1; then setfacl -m u:${TARGET_USER}:x /sys/kernel/debug /sys/kernel/debug/usb 2>/dev/null || true; setfacl -m u:${TARGET_USER}:rx /sys/kernel/debug/usb/usbmon 2>/dev/null || true; setfacl -m u:${TARGET_USER}:r /sys/kernel/debug/usb/usbmon/*u 2>/dev/null || true; fi; fi'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now ids-usbmon-permissions.service

echo "[usbmon] done."
echo "[usbmon] If direct reading still fails in the current shell, log out/in or run:"
echo "  newgrp $GROUP_NAME"
echo "[usbmon] Test:"
echo "  INTERVAL=0.5 python3 ./monitor_camera_usb_throughput.py --json --once"
