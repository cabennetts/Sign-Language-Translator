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

        // getInterp: builder.query({
        //     query: () => '/upload',
        //     // check status, make sure no error
        //     validateStatus: (response, result) => {
        //         return response.status === 200 && !result.isError
        //     },
        //     // will data be referred to in cache or need to get more?
        //     keepUnusedDataFor: 5,
            
        //     // transformResponse: responseData => {
        //     // //     const loadedUsers = responseData.map(user => {
        //     // //         user.id = user._id
        //     // //         return user
        //     // //     });
        //     // //     return usersAdapter.setAll(initialState, loadedUsers)
        //     // },
            // provides tags that can be invalidated
        //     providesTags: (result, error, arg) => {
        //     //     if (result?.ids) {
        //     //         return [
        //     //             { type: 'User', id: 'LIST' },
        //     //             ...result.ids.map(id => ({ type: 'User', id }))
        //     //         ]
        //     //     } else return [{ type: 'User', id: 'LIST' }]
        //          return [{ type: 'Test', id: 'LIST' }]
        //     }
        // }),

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
