#!/bin/sh
# Rebuild deadpool-tickets_1.0.0_all.deb from ./pkg (syncs index.html first).
set -eu
cd "$(dirname "$0")"

VERSION=$(sed -n 's/^Version: //p' pkg/DEBIAN/control)
OUT="deadpool-tickets_${VERSION}_all.deb"

cp index.html pkg/usr/share/deadpool-tickets/index.html

# Normalise permissions (dpkg-deb rejects group/world-writable control dirs).
find pkg -type d -exec chmod 0755 {} +
find pkg -type f -exec chmod 0644 {} +
chmod 0755 pkg/usr/bin/deadpool-tickets

sh -n pkg/usr/bin/deadpool-tickets
dpkg-deb --build --root-owner-group pkg "$OUT"
echo "built: $OUT"
