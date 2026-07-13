import {KeplerMap} from './KeplerMap';
import './App.css';

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Bird–Window Collision Risk — Chicago</h1>
        <p>
          Per-building relative collision risk score. Higher score = more
          hazardous to migrating birds. Model uses building geometry (Overture
          Maps), height (LiDAR), nighttime radiance (NASA Black Marble VIIRS),
          and OSM habitat proximity. Dev view: Loop + Streeterville + South
          Loop through McCormick Place. Model and website created by Alex Badgett.{' '}
          <a
            href="https://github.com/abadger6/bird-window-collision/blob/main/docs/methodology.md"
            target="_blank"
            rel="noreferrer"
          >
            Methodology
          </a>
          .
        </p>
      </header>
      <div className="app-map">
        <KeplerMap />
      </div>
    </div>
  );
}
