import {
    createSelector,
    createEntityAdapter
} from "@reduxjs/toolkit"

import { apiSlice } from "../../app/api/apiSlice"

// using EntityAdapter, we can get some normalized state here
const uploadAdapter = createEntityAdapter({})
// if initial state exists, then get it
const initialState = uploadAdapter.getInitialState()

// use apiSlice to inject endpoints into original apiSlice
export const uploadApiSlice = apiSlice.injectEndpoints({
    endpoints: builder => ({

        postVideo: builder.mutation({
            query: initialVideoData => ({
                url:'/upload',
                method: 'POST',
                body: {
                    ...initialVideoData,
                }
            }),
            
        }),
        
    }),
})

export const {
    // useGetInterpQuery,
    usePostVideoMutation,
} = uploadApiSlice
