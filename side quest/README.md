# deadpool-tickets (side quest)

Fan-made "Deadpool & Wolverine // Ticket Terminal" single-page app, packaged as a Debian package.
No real tickets are sold; not affiliated with Marvel or Disney.

## Files

- `index.html` - the page (Tailwind CDN + Google Fonts, vanilla JS; falls back to system fonts and built-in CSS offline)
- `pkg/` - package source tree (`DEBIAN/control`, launcher, `.desktop`, icon)
- `build_deb.sh` - copies `index.html` into `pkg/` and rebuilds the `.deb`

## Build

    ./build_deb.sh

## Install

    sudo dpkg -i deadpool-tickets_1.0.0_all.deb
    sudo apt-get install -f   # only if xdg-utils is missing

## Run

    deadpool-tickets

or launch "Deadpool & Wolverine Tickets" from the application menu. The launcher tries, in order:
pywebview (`python3 -c 'import webview'`), chromium / google-chrome / brave in `--app` mode,
`firefox --new-window`, then `xdg-open`.

## Uninstall

    sudo dpkg -r deadpool-tickets
