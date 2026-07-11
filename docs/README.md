# Public kepler.gl proof of concept

Drop the exported `index.html` from kepler.gl here. GitHub Pages will serve
it at `https://<user>.github.io/<repo>/`.

To update the visualization:

1. Configure the map in [kepler.gl/demo](https://kepler.gl/demo) exactly how
   you want it (load the GeoJSON, tune the 3D layer, set the camera angle).
2. Export Map → HTML. **In the Mapbox access token field, paste your own
   `pk.eyJ…` (public) token** from account.mapbox.com/access-tokens.
   Leaving the field blank makes kepler embed its own fallback `sk.`
   (secret) token — GitHub's push protection will (correctly) reject the
   push.
3. Rename the download to `index.html` and drop it here.
4. `git add docs/index.html && git commit && git push` — GitHub Pages
   redeploys automatically within ~1 min.

Recipients see the frozen view; kepler is bundled inside the HTML, no
accounts required.
