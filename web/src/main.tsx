import {createRoot} from 'react-dom/client';
import {Provider} from 'react-redux';
import 'maplibre-gl/dist/maplibre-gl.css';

import App from './App';
import {store} from './store';
import './index.css';

// No <StrictMode> wrapper — kepler.gl 3.x mishandles React 18 StrictMode's
// intentional double-mount (its reducer initialState fires twice, its modal
// singleton "already open" warns). Skipping StrictMode is the standard
// workaround; we still get React 18 concurrent features where kepler uses them.
createRoot(document.getElementById('root')!).render(
  <Provider store={store}>
    <App />
  </Provider>,
);
