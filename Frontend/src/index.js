import React from 'react';
import ReactDOM from 'react-dom/client';
import './Home.module.css';
import App from './App';
import { BrowserRouter, Routes, Route } from 'react-router-dom'
// Store
import { store } from './app/store';
// Provides global state to our app
import { Provider } from 'react-redux';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    {/* Wrap app with provider */}
    <Provider store={store}>
      <BrowserRouter>
        <Routes>
          <Route path="/*" element={<App />} />
        </Routes>
      </BrowserRouter>
    </Provider>
  </React.StrictMode>
);

