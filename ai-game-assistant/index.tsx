import React from 'react';
import ReactDOM from 'react-dom/client';
import WebUiApp from './src/WebUiApp';
import './index.css';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Could not find root element');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <WebUiApp />
  </React.StrictMode>
);
