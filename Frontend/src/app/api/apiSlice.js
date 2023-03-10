// Slice - a collection of reducer logic and actions for a single feature inside an app
    // ex: a blog might have a slice for posts and another for comments

// fetchBaseQuery is similar to axios
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'

// Create api
export const apiSlice = createApi({
    // 
    baseQuery: fetchBaseQuery({ baseUrl: ['http://localhost:3500', 'http://localhost:8000'] }),
    // used for cached data
    tagTypes: ['videoToUpload', 'postVideo', 'Test', 'Video', 'Interpretation'],
    // provide extended slices that will be attached to api slice
    endpoints: builder => ({})
})