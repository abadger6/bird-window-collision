// Redux store. Kepler.gl mounts its state under the `keplerGl` key of a
// Redux store and expects the `taskMiddleware` from `react-palm` for its
// async task machinery — that's the one hard requirement of the setup.

import {applyMiddleware, combineReducers, compose, createStore} from 'redux';
import keplerGlReducer from '@kepler.gl/reducers';
import {taskMiddleware} from 'react-palm/tasks';

// Register a token-free basemap so kepler doesn't fall back to Mapbox-tile
// URLs that require an access token. Carto's Dark Matter style is a hosted
// MapLibre/GL-JS style JSON that works out of the box. Adding it to
// mapStyles at reducer-init time is more reliable than the runtime
// addCustomMapStyle action, which has a modal step.
const CUSTOM_MAP_STYLES = {
  carto_dark: {
    id: 'carto_dark',
    label: 'Carto Dark Matter',
    url: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    icon: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/preview.png',
    layerGroups: [],
  },
  carto_positron: {
    id: 'carto_positron',
    label: 'Carto Positron',
    url: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    icon: 'https://basemaps.cartocdn.com/gl/positron-gl-style/preview.png',
    layerGroups: [],
  },
};

const customizedKeplerGlReducer = keplerGlReducer.initialState({
  mapStyle: {
    styleType: 'carto_dark',
    mapStyles: CUSTOM_MAP_STYLES,
  },
  uiState: {
    // Hide the top-right Mapbox-info panel; keeps the UI clean for the
    // public-facing view. Users can still open the side panel by clicking
    // the layer icon.
    currentModal: null,
  },
});

const reducers = combineReducers({
  // Mount point is fixed — kepler.gl actions reach into state.keplerGl.
  keplerGl: customizedKeplerGlReducer,
});

// Enable Redux DevTools when the browser extension is installed. Falls back
// to plain `compose` in production or when the extension isn't present.
type ComposeWithDevtools = typeof compose;
const composeEnhancers: ComposeWithDevtools =
  (typeof window !== 'undefined' &&
    (window as unknown as {__REDUX_DEVTOOLS_EXTENSION_COMPOSE__?: ComposeWithDevtools})
      .__REDUX_DEVTOOLS_EXTENSION_COMPOSE__) ||
  compose;

export const store = createStore(
  reducers,
  {},
  composeEnhancers(applyMiddleware(taskMiddleware)),
);

// Expose the store on window for debugging kepler state in the browser
// devtools. Guarded so it only fires in dev.
if (import.meta.env.DEV && typeof window !== 'undefined') {
  (window as unknown as {__STORE__: typeof store}).__STORE__ = store;
}

export type AppDispatch = typeof store.dispatch;
