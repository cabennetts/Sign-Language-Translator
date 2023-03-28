import React from 'react'
import { useState } from "react"
import { usePostVideoMutation } from '../features/upload/uploadApiSlice'

const UploadVideo = () => {
    // Might need this for the usePostVideoMutation() but can't get to work
    const [postVideo, {
        isLoading,
        isSuccess,
        isError,
        error
    }] = usePostVideoMutation()

    // hook for setting current file 
    const [selectedFile, setSelectedFile] = useState({});
    // form data to POST 
    const formData = new FormData()
    
    // handle change to input 
    const changeHandler = (event) => {
        setSelectedFile(event.target.files[0]);
    }

    const handleUpload = () => {
        // const fileName = selectedFile.name
        formData.append('videoToUpload', selectedFile);
        // await postVideo({ formData })
        fetch('http://localhost:3500/upload', 
        {
          method: 'POST',
          body: formData,
        }
        ).then((response) => response.json())
        .then((result) => {
          console.log('Success (EXPRESS):', result);
        })
        .catch((error) => {
          console.error('Error (EXPRESS):', error);
        })
      }

    return (
    <>
        <input type="file" id='video' name="videoToUpload" onChange={changeHandler} className='btn-logo' />
        <button onClick={handleUpload} type='submit' className='btn-green'>Upload</button>
    </>
  )
}

export default UploadVideo