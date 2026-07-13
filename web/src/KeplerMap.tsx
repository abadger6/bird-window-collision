// KeplerMap — mounts <KeplerGl>, fetches the scored buildings GeoJSON and
// the exported kepler config, and dispatches addDataToMap so the map shows
// up with the same styling as the Path 1 HTML export.
//
// The dataId in the config JSON (`oqp2ik`, extracted from the exported HTML)
// MUST match the dataset id we register — otherwise kepler renders no layer.

import {useEffect, useRef, useState} from 'react';
import {useDispatch} from 'react-redux';
import {addDataToMap} from '@kepler.gl/actions';
import {processGeojson} from '@kepler.gl/processors';
import {KeplerGlSchema} from '@kepler.gl/schemas';
import KeplerGl from '@kepler.gl/components';

import type {AppDispatch} from './store';

// dataIds referenced from kepler-config.json — must match layer.config.dataId
// there or the layer won't bind to any data and won't render.
const BLD_DATASET_ID = 'oqp2ik';
const BLD_DATASET_LABEL = 'Chicago buildings — collision risk (dev bbox)';
const CBCM_DATASET_ID = 'cbcm';
const CBCM_DATASET_LABEL = 'CBCM observations 2018–2021';

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
        const [bldResp, cbcmResp, configResp] = await Promise.all([
          fetch(`${base}geojson/chicago_buildings_dev_scored.geojson`),
          fetch(`${base}geojson/cbcm_points.geojson`),
          fetch(`${base}kepler-config.json`),
        ]);
        if (!bldResp.ok)   throw new Error(`buildings GeoJSON ${bldResp.status}`);
        if (!cbcmResp.ok)  throw new Error(`CBCM GeoJSON ${cbcmResp.status}`);
        if (!configResp.ok) throw new Error(`config ${configResp.status}`);

        const [bldGeojson, cbcmGeojson, configJson] = await Promise.all([
          bldResp.json(),
          cbcmResp.json(),
          configResp.json(),
        ]);

        if (cancelled) return;

        const bldProcessed  = processGeojson(bldGeojson);
        const cbcmProcessed = processGeojson(cbcmGeojson);
        if (!bldProcessed)  throw new Error('processGeojson(buildings) returned null');
        if (!cbcmProcessed) throw new Error('processGeojson(cbcm) returned null');

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
            // Order matters: buildings first so its centerMap wins.
            datasets: [
              {
                info: {id: BLD_DATASET_ID,  label: BLD_DATASET_LABEL},
                data: bldProcessed,
              },
              {
                info: {id: CBCM_DATASET_ID, label: CBCM_DATASET_LABEL},
                data: cbcmProcessed,
              },
            ],
            options: {centerMap: true, readOnly: false},
            config: parsedConfig,
          }),
        );
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
