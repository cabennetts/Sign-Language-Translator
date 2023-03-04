// Redux Store & Redux are used interchangeably 
// - both stand for a container for js apps
// - stores the entire state of the app in an immutable object tree

// Start the store
import { configureStore } from "@reduxjs/toolkit";
// Pull apiSlice into store
import { apiSlice } from './api/apiSlice';

// create the store
export const store = configureStore({
    // 
    reducer: {
        // dynamically referring to apiSlice using reducerPath
        [apiSlice.reducerPath]: apiSlice.reducer,
    },
    // Get default mw and add apiSlice mw
    middleware: getDefaultMiddleware =>
        getDefaultMiddleware().concat(apiSlice.middleware),
    devTools: true
})