// KeplerMap — mounts <KeplerGl>, fetches the scored buildings GeoJSON and
// the exported kepler config, and dispatches addDataToMap so the map shows
// up with the same styling as the Path 1 HTML export.
//
// The dataId in the config JSON (`oqp2ik`, extracted from the exported HTML)
// MUST match the dataset id we register — otherwise kepler renders no layer.

import {useEffect, useRef, useState} from 'react';
import {useDispatch} from 'react-redux';
import {addDataToMap, toggleSidePanel} from '@kepler.gl/actions';
import {processGeojson} from '@kepler.gl/processors';
import {KeplerGlSchema} from '@kepler.gl/schemas';
import KeplerGl from '@kepler.gl/components';

import type {AppDispatch} from './store';

// Same dataId used in the exported kepler-config.json. Rename in both places
// if you export a fresh config.
const DATASET_ID = 'oqp2ik';
const DATASET_LABEL = 'Chicago buildings — collision risk (dev bbox)';

type Size = {width: number; height: number};

function useSize(ref: React.RefObject<HTMLDivElement | null>): Size {
  const [size, setSize] = useState<Size>({width: 800, height: 600});
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const update = () => setSize({width: el.clientWidth, height: el.clientHeight});
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

export function KeplerMap() {
  const dispatch = useDispatch<AppDispatch>();
  const containerRef = useRef<HTMLDivElement>(null);
  const size = useSize(containerRef);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const base = import.meta.env.BASE_URL;
    (async () => {
      try {
        const [geojsonResp, configResp] = await Promise.all([
          fetch(`${base}geojson/chicago_buildings_dev_scored.geojson`),
          fetch(`${base}kepler-config.json`),
        ]);
        if (!geojsonResp.ok) throw new Error(`GeoJSON ${geojsonResp.status}`);
        if (!configResp.ok) throw new Error(`config ${configResp.status}`);

        const geojson = await geojsonResp.json();
        const configJson = await configResp.json();

        if (cancelled) return;

        const processed = processGeojson(geojson);
        if (!processed) throw new Error('processGeojson returned null');

        // The exported kepler config carries the Mapbox-token-required
        // "voyager" style. Force our token-free Carto Dark Matter style
        // in its place — otherwise the basemap silently fails to render.
        const configWithSafeBasemap = {
          ...configJson,
          config: {
            ...configJson.config,
            mapStyle: {
              ...(configJson.config?.mapStyle ?? {}),
              styleType: 'carto_dark',
            },
          },
        };

        const parsedConfig = KeplerGlSchema.parseSavedConfig(configWithSafeBasemap);

        dispatch(
          addDataToMap({
            datasets: {
              info: {id: DATASET_ID, label: DATASET_LABEL},
              data: processed,
            },
            options: {centerMap: true, readOnly: false},
            config: parsedConfig,
          }),
        );

        // Collapse the side panel to just its icon strip so first-load
        // impression is map-forward. Users can click the strip to expand
        // the Layers / Filters / Interactions panels back out.
        // Dispatched after addDataToMap because kepler resets activeSidePanel
        // to 'layer' when the datasets change.
        dispatch(toggleSidePanel(''));
      } catch (err) {
        console.error('Failed to load map data', err);
        if (!cancelled) setLoadError(String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [dispatch]);

  return (
    <div ref={containerRef} style={{width: '100%', height: '100%', position: 'relative'}}>
      {loadError && (
        <div style={{padding: 16, color: '#c00', fontFamily: 'sans-serif'}}>
          Map load failed: {loadError}
        </div>
      )}
      <KeplerGl
        id="birdcollision"
        width={size.width}
        height={size.height}
        // Empty token — we use MapLibre's default OSM-derived styles rather
        // than Mapbox. If you later want Mapbox styles, paste a pk.* token here.
        mapboxApiAccessToken=""
      />
    </div>
  );
}
